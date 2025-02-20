import whisperx
import gc 
import os
import torch
import argparse
from multiprocessing import Process
from whisperx.utils import get_writer
import glob

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def run_transcription(audio_filename, audio_folder):
    """
    Transcribes an audio file, aligns the transcription, and assigns speaker labels.
    
    Parameters:
    audio_filename (str): The name of the audio file to be transcribed.
    audio_folder (str): The folder containing the audio file.

    The function performs the following steps:
    1. Loads the Whisper model and transcribes the audio file.
    2. Aligns the transcription output.
    3. Performs speaker diarization to assign speaker labels to each word in the transcription.
    4. Saves the final transcription with speaker labels into a VTT file.
    
    Note:
    - The function uses GPU for processing if available.
    - The Hugging Face token is required for the diarization model.
    - The function manages GPU memory by deleting models and clearing the cache when necessary.
    """

    # -----------------------------------------
    # Parameters
    # -----------------------------------------
    device          = "cuda"
    token_hf        = os.getenv("HuggingFaceToken")
    batch_size      = 16                                        # reduce if low on GPU mem
    compute_type    = "float16"                                 # change to "int8" if low on GPU mem (may reduce accuracy)
    language_code   = "en"                                      # change to "es" for Spanish as example
    audio_file      = os.path.join(audio_folder, audio_filename)  # audio_filename passed as an argument.

    # -----------------------------------------
    # Set the Model
    # -----------------------------------------  
    # Transcribe with original whisper (batched)
    print("############### Set the Model                        ###############")        
    model = whisperx.load_model("large-v2", device, compute_type=compute_type, language=language_code)

    # save model to local path (optional)
    # model_dir = "/path/"
    # model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)



    # -----------------------------------------
    # Transcribe the Audio File
    # -----------------------------------------
    print("############### Start Transcription                  ###############")     
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    # print(result["segments"]) # before alignment

    # delete model if low on GPU resources
    if torch.cuda.memory_allocated(device) > 0.8 * torch.cuda.max_memory_reserved(device):
        gc.collect()
        torch.cuda.empty_cache()
    del model
    print("############### End Transcription                    ###############")        


    # -----------------------------------------
    # Align whisper output
    # -----------------------------------------
    print("############### Aligninment Started                  ###############")    
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    
    # delete model if low on GPU resources
    if torch.cuda.memory_reserved(device) > 0.8 * torch.cuda.max_memory_reserved(device):
        gc.collect()
        torch.cuda.empty_cache()
    del model_a

    # print(aligned_result["segments"]) # after alignment
    print("############### Alignment Finished                   ###############")    




    # -----------------------------------------
    # Assign speaker labels
    # -----------------------------------------
    print("############### Start Adding Speaker ID              ###############")

    # Instantiate the diarization model with the Hugging Face token for authentication
    diarize_model = whisperx.DiarizationPipeline(use_auth_token = token_hf , device=device)
        
    # Perform speaker diarization on the audio
    diarize_segments = diarize_model(audio)
   
    # add min/max number of speakers if known
    diarize_model(audio, min_speakers=1, max_speakers=1)

    print("############### diarized segments                    ###############")
    # print(diarize_segments)    


    # assign speaker labels to each word in the transcription
    diarize_result = whisperx.assign_word_speakers(diarize_segments,aligned_result) 

    print("############### diarized results                     ###############")
    # print(diarize_result["segments"]) # segments are now assigned speaker IDs

    print("############### End Adding Speaker ID                ###############")




    # -----------------------------------------
    # Custom function to flatten nested dictionaries
    # -----------------------------------------
    def flatten_dict(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    # Flatten the diarize_result before writing
    flattened_diarize_result = flatten_dict(diarize_result)

    # Save results into file
    print("############### Start Saving Transcription to File   ###############")
    flattened_diarize_result["language"] = result["language"]
    tsv_writer = get_writer("tsv", audio_folder)
    tsv_writer(flattened_diarize_result, audio_file, {})
    print("############### End Saving Transcription to File     ###############")        
 
 

# -----------------------------------------
# Start Process - Calls Run Process
# -----------------------------------------
def start_transcription_process(audio_folder):
    print("############### Start transcription_process          ###############")   
    flac_files = glob.glob(os.path.join(audio_folder, "*.flac"))
    for audio_filename in flac_files:
        p = Process(target=run_transcription, args=(audio_filename, audio_folder))
        p.start()
        p.join()
    print("############### End transcription_process            ###############")   



# -----------------------------------------
# Main Process - Calls Start Process
# -----------------------------------------
if __name__ == '__main__':
    print("############### Start Main                           ###############") 
    parser = argparse.ArgumentParser(description='Transcribe audio files in a folder.')
    parser.add_argument('audio_folder', type=str, help='The folder containing the audio files to transcribe')
    args = parser.parse_args()
    start_transcription_process(args.audio_folder)
    print("############### End Main                             ###############")