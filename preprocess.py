import argparse
import numpy as np
import os
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow_hub as hub
import tensorflow as tf


parser = argparse.ArgumentParser()

#transcript file
parser.add_argument(
    'transcriptfile', default='labels_processed/all_transcripts.csv', type=str, nargs="?", help="All transcriptions (cleaned)")

#Train split file
parser.add_argument(
    '--subsetfile', default='labels/full_test_split.csv', type=str)

#Directory that contains transcript files 
parser.add_argument('--transcriptdir', type=str, default='labels_processed')

#Output to store embeddings
parser.add_argument('-o', '--output', type=str,
                    default='train_elmo.ark', help='feature output')

parser.add_argument('-w', type=int, default=4, help="Worker count")

parser.add_argument('--filterlen', default=0, type=int)

parser.add_argument('--filterby', type=str, default='Participant')
args = parser.parse_args()

# Extracting features for the Participant IDs
subset_df = pd.read_csv(args.subsetfile)
speakers = subset_df['Participant_ID'].values
print(speakers)
#print(subset_df.head())

X=pd.DataFrame(columns=['speaker','value'])
for speaker in tqdm(speakers):
    # Process transcript first to get start_end
    transcript_file = glob(os.path.join(
        args.transcriptdir, str(speaker)) + '*TRANSCRIPT.csv')[0]
    transcript_df = pd.read_csv(transcript_file, sep='\t')
    transcript_df.value = transcript_df.value.str.strip()
    transcript_df.dropna(inplace=True)
    transcript_df = transcript_df[transcript_df.value.str.split().apply(
        len) > args.filterlen]
    # Filter for participant only
    if args.filterby:
        transcript_df = transcript_df[transcript_df.speaker ==
                                        args.filterby]
    
    X_temp=pd.DataFrame({'speaker':speaker,'value':transcript_df.groupby('speaker')['value'].apply(' '.join)})
    print(X_temp)
    X=X.append(X_temp)   


X['PHQ8']=subset_df['PHQ8_Binary'].values
print(X)
with open('test_split.csv','a') as f:
    X.to_csv(f,index=False)
