import pandas as pd
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator


def export_to_csv(log_dir, output_path):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    all_data = []
    for tag in ea.Tags()["scalars"]:
        for event in ea.Scalars(tag):
            all_data.append({"tag": tag, "step": event.step, "value": event.value})

    df = pd.DataFrame(all_data)
    df.to_csv(output_path, index=False)
    return df

log_dir = Path(__file__).parent
df = export_to_csv(str(log_dir), "training_logs.csv")
