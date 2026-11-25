"""
Baseline Implementations (B1 to B6) and FedZT-Edge Treatment Orchestrator
Executes B1 (Centralized), B2 (IID FedAvg), B3 (Non-IID FedAvg),
B4 (Non-IID FedProx), B5 (Non-IID FL + GMM Filter), B6 (Local Wasm-only),
and FedZT-Edge (Full Five-Layer Stack).
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

try:
    from .layer1_wasm_pep import WasmPolicyEnforcementPoint
    from .layer2_federated import FederatedServer, EdgeIDSModel, get_parameters_vector, set_parameters_vector
    from .layer3_drl_pdp import DRLTrustAgent
    from .layer4_crypto import CryptoEnvelope
    from .layer5_metrics import MetricsTracker
    from .preprocessing import partition_dirichlet_non_iid
except (ImportError, ValueError):
    from layer1_wasm_pep import WasmPolicyEnforcementPoint
    from layer2_federated import FederatedServer, EdgeIDSModel, get_parameters_vector, set_parameters_vector
    from layer3_drl_pdp import DRLTrustAgent
    from layer4_crypto import CryptoEnvelope
    from layer5_metrics import MetricsTracker
    from preprocessing import partition_dirichlet_non_iid

def evaluate_predictions(y_true, y_pred, y_probs=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_probs if y_probs is not None else y_pred)
    except Exception:
        auc = 0.5
    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "roc_auc": float(auc)
    }

def run_centralized_b1(X_train, y_train, X_test, y_test, epochs=15, batch_size=32, lr=0.001):
    """
    B1: Centralized training on pooled data (Upper Bound).
    """
    input_dim = X_train.shape[1]
    model = EdgeIDSModel(input_dim=input_dim)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.LongTensor(y_train)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    start_time = time.perf_counter()
    for ep in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
    train_time = time.perf_counter() - start_time
    
    model.eval()
    with torch.no_grad():
        test_out = model(torch.FloatTensor(X_test))
        probs = torch.softmax(test_out, dim=1)[:, 1].numpy()
        preds = np.argmax(test_out.numpy(), axis=1)
        
    metrics = evaluate_predictions(y_test, preds, probs)
    metrics["train_time_sec"] = train_time
    metrics["baseline_id"] = "B1"
    metrics["baseline_name"] = "Centralized (Pooled)"
    return metrics

def run_local_wasm_only_b6(X_train, y_train, X_test, y_test, num_clients=10, alpha=0.5, epochs=15, seed=42):
    """
    B6: Local Wasm-only IDS on each device (No FL, No DRL).
    Each client trains only on its own local data shard in isolation.
    """
    shards = partition_dirichlet_non_iid(X_train, y_train, num_clients=num_clients, alpha=alpha, seed=seed)
    client_f1s = []
    client_accs = []
    client_precs = []
    client_recs = []
    client_aucs = []
    total_fuel = 0
    
    start_time = time.perf_counter()
    for c in range(num_clients):
        X_c, y_c = shards[c]
        pep = WasmPolicyEnforcementPoint(client_id=c, execution_mode="wasm")
        
        # Local model
        m = EdgeIDSModel(input_dim=X_c.shape[1])
        m.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(m.parameters(), lr=0.001)
        
        X_t = torch.FloatTensor(X_c)
        y_t = torch.LongTensor(y_c)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_t, y_t), 
            batch_size=min(32, len(X_c)), 
            shuffle=True
        )
        
        for ep in range(epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                out = m(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                
        m.eval()
        # Evaluate on test set
        res = pep.run_inference(m, X_test)
        preds = (res["anomaly_probs"] >= 0.5).astype(int)
        client_accs.append(accuracy_score(y_test, preds))
        client_f1s.append(f1_score(y_test, preds, zero_division=0))
        client_precs.append(precision_score(y_test, preds, zero_division=0))
        client_recs.append(recall_score(y_test, preds, zero_division=0))
        try:
            client_aucs.append(roc_auc_score(y_test, res["anomaly_probs"]))
        except Exception:
            client_aucs.append(0.5)
        total_fuel += res["fuel_consumed"]
        
    duration = time.perf_counter() - start_time
    return {
        "baseline_id": "B6",
        "baseline_name": "Local Wasm-Only (Isolated)",
        "accuracy": float(np.mean(client_accs)),
        "accuracy_std": float(np.std(client_accs)),
        "f1": float(np.mean(client_f1s)),
        "f1_std": float(np.std(client_f1s)),
        "precision": float(np.mean(client_precs)),
        "recall": float(np.mean(client_recs)),
        "roc_auc": float(np.mean(client_aucs)),
        "train_time_sec": duration,
        "total_fuel": total_fuel
    }

def run_federated_pipeline(X_train, y_train, X_test, y_test,
                           config_id="Treatment",
                           num_clients=10,
                           alpha=0.5,
                           rounds=15,
                           local_epochs=3,
                           aggregation_method="FedProx",
                           fedprox_mu=0.1,
                           use_defense=True,
                           malicious_rate=0.0,
                           use_drl=True,
                           use_crypto=True,
                           hardware_tier="Raspberry_Pi_4",
                           seed=42):
    """
    Unified runner covering:
    - B2: IID FedAvg (alpha=inf, no defense, no DRL)
    - B3: Non-IID FedAvg (alpha=0.5, no defense, no DRL)
    - B4: Non-IID FedProx (alpha=0.5, mu=0.1, no defense, no DRL)
    - B5: Non-IID FL + GMM Filter (alpha=0.5, mu=0.1, defense=True, no DRL)
    - Treatment: FedZT-Edge (Non-IID FedProx + GMM Defense + DRL Trust Agent + Wasm PEP + Crypto + Instrumentation)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 1. Shard data
    shards = partition_dirichlet_non_iid(X_train, y_train, num_clients=num_clients, alpha=alpha, seed=seed)
    
    # Assign malicious clients
    num_malicious = int(round(num_clients * malicious_rate))
    malicious_ids = set(np.random.choice(num_clients, size=num_malicious, replace=False))
    
    # 2. Initialize Infrastructure
    server = FederatedServer(
        input_dim=X_train.shape[1],
        aggregation_method=aggregation_method,
        fedprox_mu=fedprox_mu,
        use_defense=use_defense,
        root_data=(X_train[:100], y_train[:100])
    )
    
    peps = [WasmPolicyEnforcementPoint(client_id=c, execution_mode="wasm") for c in range(num_clients)]
    crypto = CryptoEnvelope(enabled=use_crypto)
    metrics = MetricsTracker(hardware_tier=hardware_tier)
    drl_agent = DRLTrustAgent() if use_drl else None
    
    round_f1_history = []
    
    # Federated Rounds
    start_total_time = time.perf_counter()
    for r in range(rounds):
        round_start = time.perf_counter()
        current_weights = get_parameters_vector(server.global_model)
        
        # Simulated Crypto Envelope for Global Model Broadcast
        broadcast_env = crypto.pack_and_encrypt({"round": r, "weights": current_weights.tolist()})
        metrics.record_network_transfer(len(broadcast_env["ciphertext"]) * num_clients)
        
        client_updates = []
        for c in range(num_clients):
            is_mal = (c in malicious_ids)
            X_c, y_c = shards[c]
            
            # Local training on edge client
            client_res = server.train_client_local(
                client_id=c,
                model_state=current_weights,
                X_client=X_c,
                y_client=y_c,
                epochs=local_epochs,
                lr=0.001,
                is_malicious=is_mal
            )
            
            # Encrypt local update payload (Layer 4)
            update_env = crypto.pack_and_encrypt({"client_id": c, "delta": client_res["delta"].tolist()})
            metrics.record_network_transfer(len(update_env["ciphertext"]))
            client_updates.append(client_res)
            
        # Server Aggregation with GMM+K-means Defense (Layer 2)
        new_weights, filter_metrics = server.aggregate(client_updates)
        
        # Test Global Model on Test Split
        eval_metrics = server.evaluate_global(X_test, y_test)
        round_f1_history.append(eval_metrics["f1"])
        
        round_duration = time.perf_counter() - round_start
        metrics.record_round_time(round_duration)
        
    total_time = time.perf_counter() - start_total_time
    
    # 3. Post-Training DRL Zero-Trust Mitigation Loop (Layer 3)
    drl_rewards = []
    drl_action_dist = {0: 0, 1: 0, 2: 0, 3: 0}
    
    if use_drl and drl_agent is not None:
        # Evaluate 500 test samples through DRL Policy Decision Point
        test_sub_idx = np.random.choice(len(y_test), size=min(500, len(y_test)), replace=False)
        sub_X = X_test[test_sub_idx]
        sub_y = y_test[test_sub_idx]
        
        pep_ref = peps[0]
        inf_res = pep_ref.run_inference(server.global_model, sub_X)
        probs = inf_res["anomaly_probs"]
        
        curr_trust = 1.0
        prev_action = 0
        
        for idx in range(len(sub_X)):
            anomaly_prob = float(probs[idx])
            flow_entropy = float(np.random.uniform(0.1, 0.9))
            resource_load = float(np.random.uniform(0.2, 0.8))
            
            # S_t = [anomaly_prob, trust_score, flow_entropy, prev_action, resource_load]
            state = np.array([anomaly_prob, curr_trust, flow_entropy, float(prev_action), resource_load])
            action = drl_agent.select_action(state, evaluate=True)
            drl_action_dist[action] += 1
            
            reward = drl_agent.compute_reward(state, action, sub_y[idx])
            drl_rewards.append(reward)
            
            # Trust update rule: decay on high anomaly or enforcement, recover on benign allow
            if action in [2, 3]:
                curr_trust = max(0.05, curr_trust - 0.25)
            elif action == 0 and anomaly_prob < 0.3:
                curr_trust = min(1.0, curr_trust + 0.05)
                
            prev_action = action
            metrics.record_inference(inf_res["latency_ms"] / len(sub_X), inf_res["fuel_consumed"] / len(sub_X))
            
    summary_metrics = metrics.summary()
    final_eval = server.evaluate_global(X_test, y_test)
    
    result = {
        "baseline_id": config_id,
        "f1": final_eval["f1"],
        "accuracy": final_eval["accuracy"],
        "precision": final_eval["precision"],
        "recall": final_eval["recall"],
        "roc_auc": final_eval["roc_auc"],
        "round_f1_history": round_f1_history,
        "total_time_sec": total_time,
        "mb_transferred": summary_metrics["mb_transferred"],
        "energy_joules_per_round": summary_metrics["energy_joules_per_round"],
        "latency_stats": summary_metrics["latency"],
        "drl_avg_reward": float(np.mean(drl_rewards)) if drl_rewards else None,
        "drl_action_distribution": drl_action_dist,
        "filter_metrics_last_round": server.filter_metrics_history[-1] if server.filter_metrics_history else None
    }
    return result
