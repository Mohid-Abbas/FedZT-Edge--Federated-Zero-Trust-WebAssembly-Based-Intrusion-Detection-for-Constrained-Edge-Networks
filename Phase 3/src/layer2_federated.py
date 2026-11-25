"""
Layer 2: Federated Learning Engine with Poisoning Defence (FedZT-Edge)
Implements FedAvg, FedProx, FedNova, Adversarial Label-Flipping Attacks,
and GMM + K-Means Statistical Update Filtering.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

# Lightweight MLP Edge Detector Model
class EdgeIDSModel(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=32, num_classes=2):
        super(EdgeIDSModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

def get_parameters_vector(model):
    """Flattens all trainable parameters into a 1D numpy array."""
    params = []
    for param in model.parameters():
        params.append(param.detach().cpu().numpy().flatten())
    return np.concatenate(params)

def set_parameters_vector(model, vector):
    """Sets model parameters from a flattened 1D numpy array."""
    offset = 0
    with torch.no_grad():
        for param in model.parameters():
            num_elements = param.numel()
            param_shape = param.shape
            chunk = vector[offset:offset + num_elements]
            param.copy_(torch.from_numpy(chunk.reshape(param_shape)))
            offset += num_elements

class FederatedServer:
    def __init__(self, input_dim=20, aggregation_method="FedAvg", fedprox_mu=0.1, use_defense=True, root_data=None):
        self.global_model = EdgeIDSModel(input_dim=input_dim)
        self.aggregation_method = aggregation_method  # "FedAvg", "FedProx", "FedNova", "FLTrust"
        self.fedprox_mu = fedprox_mu
        self.use_defense = use_defense
        self.root_data = root_data
        self.round_history = []
        self.filter_metrics_history = []

    def train_client_local(self, client_id, model_state, X_client, y_client, 
                           epochs=3, batch_size=32, lr=0.001, is_malicious=False):
        """
        Executes local training on an edge client shard.
        If is_malicious=True, conducts label-flipping attack (flipping 0 <-> 1).
        """
        local_model = EdgeIDSModel(input_dim=X_client.shape[1])
        set_parameters_vector(local_model, model_state)
        local_model.train()
        
        y_train_client = y_client.copy()
        if is_malicious:
            # Targeted label-flipping attack: invert benign and attack labels
            y_train_client = 1 - y_train_client
            
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(local_model.parameters(), lr=lr)
        
        X_tensor = torch.FloatTensor(X_client)
        y_tensor = torch.LongTensor(y_train_client)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        global_vec = model_state.copy()
        global_tensor = torch.FloatTensor(global_vec)
        
        for ep in range(epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = local_model(batch_x)
                loss = criterion(outputs, batch_y)
                
                # Add proximal regularization term for FedProx
                if self.aggregation_method == "FedProx" and self.fedprox_mu > 0:
                    local_vec = []
                    for p in local_model.parameters():
                        local_vec.append(p.view(-1))
                    local_tensor = torch.cat(local_vec)
                    prox_loss = (self.fedprox_mu / 2.0) * torch.norm(local_tensor - global_tensor) ** 2
                    loss = loss + prox_loss
                    
                loss.backward()
                optimizer.step()
                
        local_vec = get_parameters_vector(local_model)
        update_delta = local_vec - global_vec
        
        return {
            "client_id": client_id,
            "weights": local_vec,
            "delta": update_delta,
            "num_samples": len(X_client),
            "is_malicious": is_malicious
        }

    def filter_updates_gmm_kmeans(self, client_updates):
        """
        Step 5: GMM + K-Means Statistical Update Filter (Djemaa et al. 2026).
        Two-stage defence against label-flipping gradient poisoning:
          Stage A — Norm-ratio outlier rejection:
                    Uses a 20%-trimmed mean as the robust norm reference.
                    Flags updates whose norm exceeds 4× the trimmed mean
                    (inflated gradients typical of full label-flip attacks).
          Stage B — GMM bimodal clustering on [norm_ratio, pairwise_avg_cosine_dist]:
                    Fits a 2-component GMM. The cluster with higher average
                    pairwise cosine distance is flagged if it is a strict minority
                    (<= 30% of clients, Djemaa guard) AND its avg cosine dist
                    exceeds the other cluster by > 0.4 (prevents false positives
                    from normal non-IID gradient variance).
        Reports filter_precision, filter_recall, benign_preservation.
        """
        if len(client_updates) <= 2 or not self.use_defense:
            return client_updates, {"filter_precision": 1.0, "filter_recall": 0.0, "benign_preservation": 1.0}

        deltas = np.array([u["delta"] for u in client_updates])
        ground_truth = np.array([u["is_malicious"] for u in client_updates])
        n = len(client_updates)

        norms = np.linalg.norm(deltas, axis=1)

        # Robust reference: 20%-trimmed mean norm
        sorted_norm_idx = np.argsort(norms)
        trim_k = max(1, int(n * 0.2))
        trimmed_idx = sorted_norm_idx[trim_k:-trim_k] if trim_k < n // 2 else sorted_norm_idx
        trimmed_norm_mean = np.mean(norms[trimmed_idx]) + 1e-9

        # Stage A: flag extreme norm outliers (> 4× trimmed mean)
        norm_ratio = norms / trimmed_norm_mean
        flagged = norm_ratio > 4.0

        # Stage B: GMM on [norm_ratio, pairwise avg cosine distance]
        if n >= 6:
            try:
                norms_safe = norms + 1e-9
                normed = deltas / norms_safe[:, None]
                cos_sim_matrix = normed @ normed.T
                avg_pairwise_cos_dist = np.array([
                    np.mean(1.0 - np.delete(cos_sim_matrix[i], i))
                    for i in range(n)
                ])

                features = np.column_stack([np.clip(norm_ratio, 0, 10), avg_pairwise_cos_dist])
                gmm = GaussianMixture(n_components=2, random_state=42, max_iter=200)
                gm_labels = gmm.fit_predict(features)

                c0_idx = np.where(gm_labels == 0)[0]
                c1_idx = np.where(gm_labels == 1)[0]
                c0_cos = np.mean(avg_pairwise_cos_dist[c0_idx])
                c1_cos = np.mean(avg_pairwise_cos_dist[c1_idx])

                # Djemaa guard: suspect cluster must be strict minority (<= 30%)
                # and must have meaningfully higher cosine distance (> 0.4 gap)
                max_suspect = max(1, int(n * 0.3))
                if c1_cos > c0_cos + 0.4 and len(c1_idx) <= max_suspect:
                    flagged[c1_idx] = True
                elif c0_cos > c1_cos + 0.4 and len(c0_idx) <= max_suspect:
                    flagged[c0_idx] = True

            except Exception:
                pass

        accepted_updates = [u for i, u in enumerate(client_updates) if not flagged[i]]
        if len(accepted_updates) == 0:
            accepted_updates = [client_updates[np.argmin(norms)]]

        tp = int(np.sum(flagged & ground_truth))
        fp = int(np.sum(flagged & (~ground_truth)))
        fn = int(np.sum((~flagged) & ground_truth))
        tn = int(np.sum((~flagged) & (~ground_truth)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        benign_preservation = tn / (tn + fp) if (tn + fp) > 0 else 1.0

        return accepted_updates, {
            "filter_precision": float(precision),
            "filter_recall": float(recall),
            "benign_preservation": float(benign_preservation),
            "num_flagged": int(np.sum(flagged))
        }
            
        accepted_updates = []
        for i, u in enumerate(client_updates):
            if not flagged[i]:
                accepted_updates.append(u)
                
        if len(accepted_updates) == 0: # Safety guard against empty aggregation
            accepted_updates = [client_updates[np.argmin(norms)]]
            
        # Compute Djemaa criteria
        tp = np.sum(flagged & ground_truth)
        fp = np.sum(flagged & (~ground_truth))
        fn = np.sum((~flagged) & ground_truth)
        tn = np.sum((~flagged) & (~ground_truth))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        benign_preservation = tn / (tn + fp) if (tn + fp) > 0 else 1.0
        
        filter_metrics = {
            "filter_precision": float(precision),
            "filter_recall": float(recall),
            "benign_preservation": float(benign_preservation),
            "num_flagged": int(np.sum(flagged))
        }
        
        return accepted_updates, filter_metrics

    def aggregate(self, client_updates):
        """
        Aggregates client updates using FedAvg, FedProx, or FedNova.
        """
        filtered_updates, filter_metrics = self.filter_updates_gmm_kmeans(client_updates)
        self.filter_metrics_history.append(filter_metrics)
        
        total_samples = sum(u["num_samples"] for u in filtered_updates)
        weights_list = [u["weights"] for u in filtered_updates]
        sample_counts = [u["num_samples"] for u in filtered_updates]
        
        if self.aggregation_method in ["FedAvg", "FedProx"]:
            # Weighted coordinate-wise average
            new_global_weights = np.zeros_like(weights_list[0])
            for w, n in zip(weights_list, sample_counts):
                new_global_weights += w * (n / total_samples)
        elif self.aggregation_method == "FedNova":
            # Normalized averaging across varying local steps (Wang et al., 2020)
            deltas = [u["delta"] for u in filtered_updates]
            taus = np.array([max(1, u["num_samples"] // 32) for u in filtered_updates], dtype=float)
            tau_eff = np.sum((np.array(sample_counts) / total_samples) * taus)
            scale_ratios = np.clip(tau_eff / taus, 0.5, 2.0)
            
            current_weights = get_parameters_vector(self.global_model)
            avg_delta = np.zeros_like(deltas[0])
            for d, n, sr in zip(deltas, sample_counts, scale_ratios):
                avg_delta += (d * sr) * (n / total_samples)
            new_global_weights = current_weights + avg_delta
        elif self.aggregation_method == "FLTrust":
            # Real FLTrust (Cao et al., NDSS 2021)
            # Bootstrapped Byzantine-robust aggregation via clean root anchor
            current_weights = get_parameters_vector(self.global_model)
            if self.root_data is not None:
                X_root, y_root = self.root_data
                ref_res = self.train_client_local(
                    client_id=-1,
                    model_state=current_weights,
                    X_client=X_root,
                    y_client=y_root,
                    epochs=1,
                    lr=0.001
                )
                delta_0 = ref_res["delta"]
            else:
                deltas = np.array([u["delta"] for u in filtered_updates])
                delta_0 = np.median(deltas, axis=0)

            norm_0 = np.linalg.norm(delta_0) + 1e-9
            trust_scores = []
            normalized_updates = []
            for u in filtered_updates:
                d_i = u["delta"]
                norm_i = np.linalg.norm(d_i) + 1e-9
                cos_sim = np.dot(d_i, delta_0) / (norm_i * norm_0)
                ts = max(0.0, float(cos_sim))
                trust_scores.append(ts)
                norm_d = (norm_0 / norm_i) * d_i
                normalized_updates.append(norm_d)

            sum_ts = sum(trust_scores)
            if sum_ts > 1e-7:
                agg_delta = np.zeros_like(current_weights)
                for ts, nd in zip(trust_scores, normalized_updates):
                    agg_delta += (ts / sum_ts) * nd
            else:
                agg_delta = delta_0

            new_global_weights = current_weights + agg_delta
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
            
        set_parameters_vector(self.global_model, new_global_weights)
        return new_global_weights, filter_metrics

    def evaluate_global(self, X_test, y_test):
        """Evaluates the global model on held-out test data."""
        self.global_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test)
            logits = self.global_model(X_tensor)
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()
            preds = np.argmax(logits.numpy(), axis=1)
            
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, zero_division=0)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        try:
            auc = roc_auc_score(y_test, probs)
        except Exception:
            auc = 0.5
            
        return {
            "accuracy": float(acc),
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "roc_auc": float(auc)
        }
