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
    model = whisperx.load_model("large-v2", device, compute_type=compute_type, language=language_code)

    # save model to local path (optional)
    # model_dir = "/path/"
    # model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)



    # -----------------------------------------
    # Transcribe the Audio File
    # -----------------------------------------
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    print(result["segments"]) # before alignment

    # delete model if low on GPU resources
    if torch.cuda.memory_allocated(device) > 0.8 * torch.cuda.max_memory_reserved(device):
        gc.collect()
        torch.cuda.empty_cache()
    del model



    # -----------------------------------------
    # Align whisper output
    # -----------------------------------------
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    print("############### after alignment ###############")    
    print(aligned_result["segments"]) # after alignment
    
    # delete model if low on GPU resources
    if torch.cuda.memory_reserved(device) > 0.8 * torch.cuda.max_memory_reserved(device):
        gc.collect()
        torch.cuda.empty_cache()
    del model_a




    # -----------------------------------------
    # Assign speaker labels
    # -----------------------------------------
    # Instantiate the diarization model with the Hugging Face token for authentication
    diarize_model = whisperx.DiarizationPipeline(use_auth_token = token_hf , device=device)
        
    # Perform speaker diarization on the audio
    diarize_segments = diarize_model(audio)
   
    # add min/max number of speakers if known
    # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
    
    # assign speaker labels to each word in the transcription
    diarize_result = whisperx.assign_word_speakers(diarize_segments,aligned_result) 

    print("############### with speaker ID ###############")
    print(diarize_segments)
    print(diarize_result["segments"]) # segments are now assigned speaker IDs




    # -----------------------------------------
    # Save results into file
    # -----------------------------------------
    diarize_result["language"] = result["language"]
    tsv_writer = get_writer("tsv", audio_folder)
    tsv_writer(diarize_result, audio_file, {})
 
    #    vtt_writer(
    #        diarize_result,
    #        audio_file,
    #        {"max_line_width": None, "max_line_count": None, "highlight_words": False},
    #    )
  
    # # Save as an TXT file
    # srt_writer = get_writer("txt", "captions/")
    # srt_writer(result, audio_file, {})

    # # Save as an SRT file
    # srt_writer = get_writer("srt", "captions/")
    # srt_writer(
    #     result,
    #     audio_file,
    #     {"max_line_width": None, "max_line_count": None, "highlight_words": False},
    # )

    # # Save as a VTT file
    # vtt_writer = get_writer("vtt", "captions/")
    # vtt_writer(
    #     result,
    #     audio_file,
    #     {"max_line_width": None, "max_line_count": None, "highlight_words": False},
    # )

    # # Save as a TSV file
    # tsv_writer = get_writer("tsv", "captions/")
    # tsv_writer(result, audio_file, {})

    # # Save as a JSON file
    # json_writer = get_writer("json", "captions/")
    # json_writer(result, audio_file, {})









# -----------------------------------------
# Start Process - Calls Run Process
# -----------------------------------------
def start_transcription_process(audio_folder):
    flac_files = glob.glob(os.path.join(audio_folder, "*.flac"))
    for audio_filename in flac_files:
        p = Process(target=run_transcription, args=(audio_filename, audio_folder))
        p.start()
        p.join()




# -----------------------------------------
# Main Process - Calls Start Process
# -----------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transcribe audio files in a folder.')
    parser.add_argument('audio_folder', type=str, help='The folder containing the audio files to transcribe')
    args = parser.parse_args()
    start_transcription_process(args.audio_folder)