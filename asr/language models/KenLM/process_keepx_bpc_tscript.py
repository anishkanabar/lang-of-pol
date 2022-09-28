import argparse
import pandas as pd
import re



def clean_text_klm(text):
    "Clean transcript text to format necessary for KenLM"
    
    # text = re.sub('<X>\s?','',text)           # removes all <x> and a space if one exists
    # text = re.sub('[\[\]]', '', text)         # removes all []
    text = re.sub(r'\s?<[^X]*?>','', text, count=0)   # removes any '<any char/characters>' except <X>
    
    text = re.sub('.*?\[.*?\].*?$', '', text)   # removes full sentence if it has a []
    # text = re.sub('.*?<.*?>.*?$', '', text)     # removes full sentence if it has a <>
    # text = re.sub('\[.*?\]', '', text)        # removes all [] and text within
    # text = re.sub('<|>', '', text)            #removes angular brackets <> alone
    text = re.sub(',', '', text)                #removes comma
    # text = re.sub('^(S\d\d\s)', '', text)       #remove prefixes made by user kkim #comma removed above
    text = "".join(c for c in text if ord(c)<128) #remove non ascii characters
    text = text.strip()
    text = text.lower()                         # keep this to the last since above regex expect uppercase

    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', help='Transcript file path')
    parser.add_argument('output_path', help='Location to create output text file')

    args = parser.parse_args()

    file_path = args.file_path
    output_folder = args.output_path

    transcript_df = pd.read_csv(file_path)

    #dropna 
    transcript_df = transcript_df.dropna(subset=['transcription'], axis='rows')  #NaN cells converted to str 'NaN' and causing errors

    #drop transcriptions by user 'kristinakim'
    transcript_df = transcript_df.drop(transcript_df[transcript_df.transcriber == 'kristinakim'].index)

    #Select transcript text as string
    df_text = transcript_df[['transcription']]

    # df_text = transcript_df[['transcription']].astype(str)

    #clean transcript text
    df_text['cleaned_transcription'] = df_text.transcription.apply(lambda x: clean_text_klm(x))

    df_text = df_text[df_text['cleaned_transcription'].ne('')]   #drops rows with zero length strings - NaNs

    output = output_folder + 'corpus.csv'

    df_text.to_csv(output)




