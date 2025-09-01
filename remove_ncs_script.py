# find all songs containing the "N.C." chord in their beatinfo files
import os
import pandas as pd
import json

nc_list = {}
chord_counts = {}

num_files = len(os.listdir("dataset/mixes"))

for i in range(1, 3001):
    file_num = f"{i:04d}"
    beatinfo_path = f"dataset/annotations/{i:04d}_beatinfo.arff"
    new_beatinfo_path = f"dataset/annotations/{i:04d}_beatinfo.csv"
    onsets_path = f"dataset/annotations/{i:04d}_onsets.arff"
    segments_path = f"dataset/annotations/{i:04d}_segments.arff"
    audio_path = f"dataset/mixes/{i:04d}_mix.flac"

    # delete onsets and segments, we don't need them
    if os.path.exists(onsets_path):
        os.remove(onsets_path)
    if os.path.exists(segments_path):
        os.remove(segments_path)
    if not os.path.exists(beatinfo_path):
        nc_list[file_num] = "No beatinfo file"
        continue

    beatinfo_df = pd.read_csv(beatinfo_path, comment="@", header=None)
    beatinfo_df.columns = ['Start time in seconds', 'Bar count', 'Quarter count', 'Chord name']

    # clean up overlapping time stamps first
    rounded_times = beatinfo_df['Start time in seconds'].round(1)
    keep_indices = ~rounded_times.duplicated(keep='last')
    beatinfo_df = beatinfo_df[keep_indices]

    # check if duplicates were removed
    if beatinfo_df.shape[0] < rounded_times.shape[0]:
        print(f"Removed duplicates from file {file_num}")

    if "'N.C.'" not in beatinfo_df['Chord name'].values and "'BASS_NOTE_EXCEPTION'" not in beatinfo_df['Chord name'].values:
        for chord in beatinfo_df['Chord name']:
            if chord not in chord_counts:
                chord_counts[chord] = 0
            chord_counts[chord] += 1
        # save cleaned beatinfo file as a csv
        beatinfo_df.to_csv(new_beatinfo_path, index=False)
        # delete the old beatinfo file
        if os.path.exists(beatinfo_path):
            os.remove(beatinfo_path)

    else:
        nc_list[file_num] = "N.C. chord found"
        # delete the beatinfo file and the audio file
        if os.path.exists(beatinfo_path):
            os.remove(beatinfo_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)

print(f"Total songs without N.C. chords: {num_files - len(nc_list)}")
print(f"Number of unique chords in songs without N.C. chords: {len(chord_counts)}")
print("Chord counts in songs without N.C. chords:")
for chord, count in chord_counts.items():
    print(f"{chord}: {count}")



