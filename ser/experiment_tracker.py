import datetime
import json

def experiment_tracker(DATA_DIR, name, epochs, batch_size, learning_rate, epoch, val_loss, val_acc):
    experiment_dir = DATA_DIR / "results" / str(name + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    hyperparams_dir = experiment_dir / "hyperparams"
    output_dir = experiment_dir / "output"

    # Create directories if they don't exist
    hyperparams_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    ## Making the hyperparameter & output files files
    hyperparameters_dict = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }

    hyperparams_file = hyperparams_dir / "hyperparameters.json"
    with open(hyperparams_file, 'w') as f:
        json.dump(hyperparameters_dict, f)

    output_data = {
        "epoch": epoch,
        "val_loss": val_loss,
        "val_acc": val_acc,
    }
    output_file = output_dir / "output.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f)

    return experiment_dir

    






