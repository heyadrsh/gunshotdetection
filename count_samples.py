import os

def count_samples():
    dataset_path = 'dataset'
    counts = {}
    total_samples = 0
    
    for gun_type in os.listdir(dataset_path):
        gun_path = os.path.join(dataset_path, gun_type)
        if os.path.isdir(gun_path):
            wav_files = [f for f in os.listdir(gun_path) if f.endswith('.wav')]
            counts[gun_type] = len(wav_files)
            total_samples += len(wav_files)
    
    print("\nGun Type Sample Counts:")
    print("-" * 30)
    for gun_type, count in counts.items():
        print(f"{gun_type}: {count} samples")
    print("-" * 30)
    print(f"Total samples: {total_samples}")

if __name__ == "__main__":
    count_samples() 