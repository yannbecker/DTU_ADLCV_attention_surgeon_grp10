import torch
import torch.nn as nn
from codecarbon import EmissionsTracker
import os
import argparse
import pandas as pd

# Import your existing classes/functions
from classification import DinoClassifier, validate, get_loaders
from physical_pruning import PhysicalPruner 

def evaluate_and_track(model, loader, device, model_name, output_dir, step_num):
    """Runs inference and tracks CO2 for a specific pruning step."""
    print(f"\n>>> [Step {step_num}] Measuring Carbon Footprint for: {model_name} on {device}")
    
    tracker = EmissionsTracker(
        project_name=f"DINOv2_Step_{step_num}",
        output_dir=output_dir,
        output_file="all_steps_emissions.csv",
        log_level="error"
    )
    
    tracker.start()
    try:
        model.eval()
        with torch.no_grad():
            # Ensure model is on the correct device for validation
            _, accuracy = validate(model, loader, nn.CrossEntropyLoss(), device)
    finally:
        emissions = tracker.stop()
        
    print(f"    Accuracy: {accuracy:.2f}% | CO2: {emissions:.6f} kg")
    return {"step": step_num, "accuracy": accuracy, "emissions_kg": emissions}

def main(args):
    # Determine the device based on user argument
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    results = []

    # 1. Load the Test Data
    _, test_loader, num_classes = get_loaders(
        args.dataset, args.data_dir, args.batch_size, num_workers=args.num_workers
    )

    # 2. Load the Sequence Tensor
    if not os.path.exists(args.sequence_path):
        raise FileNotFoundError(f"Sequence tensor not found at {args.sequence_path}")
    sequence_tensor = torch.load(args.sequence_path, map_location=device)

    # 3. Load the Unpruned Base Model
    base_model = DinoClassifier(device=device, num_classes=num_classes).to(device)
    base_checkpoint = torch.load(args.full_weights, map_location=device)
    base_model.load_state_dict(base_checkpoint.get('model_state_dict', base_checkpoint))

    # --- PART 1: BASELINE ---
    res = evaluate_and_track(base_model, test_loader, device, "Full_Model", args.save_dir, 0)
    results.append(res)

    # --- PART 2: SEQUENTIAL PRUNING STUDY ---
    print("\n[Sequential Study] Starting Step-by-Step Physical Pruning...")
    
    model_generator = PhysicalPruner.yield_sequential_models(base_model, sequence_tensor)
    
    for step, pruned_model in model_generator:
        # Crucial: Move the newly generated model to the requested device
        pruned_model.to(device)
        
        res = evaluate_and_track(
            pruned_model, 
            test_loader, 
            device, 
            f"Pruned_Step_{step}", 
            args.save_dir, 
            step
        )
        results.append(res)

        # Cleanup
        del pruned_model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # 4. Save Final Summary
    df = pd.DataFrame(results)
    summary_path = os.path.join(args.save_dir, "pruning_carbon_summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"\n[Finished] Summary saved to {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_weights", type=str, required=True)
    parser.add_argument("--sequence_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="imagenet100")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="./emissions_study")
    
    # New Device Argument
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda or cpu)"
    )
    
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    main(args)