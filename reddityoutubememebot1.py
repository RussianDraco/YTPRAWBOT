import ffmpeg
import praw
import os
import numpy as np
import urllib
from urllib.request import urlopen, HTTPError, URLError
import cv2
import re
import shutil
import glob
import PIL
from PIL import Image, ImageDraw, ImageFont
import textwrap
import moviepy.editor as mpe
from moviepy.editor import VideoFileClip, concatenate_videoclips
from mutagen.mp3 import MP3
from pydub import AudioSegment, silence
import random
import pyttsx3
from dotenv import load_dotenv
import io
import audioread
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import subprocess
import wave
#from titlecompiler import get_mp3f, get_wavf
from pathlib import Path
from converter import Converter
from title_cleaner import clean_title

load_dotenv()

channel_name = 'MemesGeek'
video_name = ''
img_filetype = '.jpg'
vid_filetype = '.avi'
folder_path = ''
log_path = ''
subreddit_name = input("Enter reddit/subreddit name:")
post_option = input("Would you like to post video (y or n)?")
if not post_option == 'y' or not post_option == 'n':
    post_option = input("Would you like to post video (y or n)?")
#run_loc = input("Are you running in cmd?")
origin = r'C:/Users/mathn/OneDrive/Desktop/YoutubeMemePyProject/'
black_rect = r'C:/Users/mathn/OneDrive/Desktop/YoutubeMemePyProject/black_rect.jpg'
music_folder = r'C:/Users/mathn'
musics = []
video_status = 'private'
frame_length = []
thumbnail_path = ''


def replace_line(file_name, line_num, text):
    if os.path.exists(file_name):
        lines = open(file_name, 'r').readlines()
        lines[line_num] = text
        out = open(file_name, 'w')
        out.writelines(lines)
        out.close()

def mp3_to_wav(input_file, output_file):
    subprocess.call(['sox', input_file, '-e', 'mu-law', '-r', '16k', output_file, 'remix', '1,2'])

def combine_wavfiles(wav_files, wavfile_name):
    data = []

    for infile in wav_files:
        w = wave.open(infile, 'rb')
        data.append([w.getparams(), w.readframes(w.getnframes())])
        w.close()
    
    output = wave.open(wavfile_name, 'wb')
    output.setparams(data[0][0])

    for i in range(len(data)):
        output.writeframes(data[i][1])
    output.close()

def destroy_post(post_number, txtpath, image_number):
    replace_line(txtpath, post_number, '')
    n = post_number + 1
    for img in range(image_number - post_number):
        os.rename(r'C:/Users/mathn/rect' + str(n) + img_filetype, r'C:/Users/mathn/rect' + str(n - 1) + img_filetype)

        n += 1


def generate_title():
    with open(r'C:/Users/mathn/OneDrive/Desktop/YoutubeMemePyProject/vid_logbook.txt', 'r') as logbook:
        vid_log = logbook.read().lower().split('\n')
    logbook.close()

    vid_count = sum(1 for i in range(len(str(' '.join(vid_log))))
                    if str(' '.join(vid_log)).startswith(subreddit_name, i))

    video_name = subreddit_name + f' compliation number {str(vid_count + 1)}'

    video_namearray = video_name.split(' ')

    new_namearray = []

    numeric = ''

    for word in video_namearray:
        if not word.isnumeric():
            word_array = list(word)
            new_namearray.append(word_array[0].capitalize() + word[1:])
            new_namearray.append(' ')
        else:
            numeric = str(word)
        

    video_name = ''.join(new_namearray) + numeric


    return video_name    

def compile_description():
    return (f"Thank you for watching this video! - Sub and Like? - Source: {subreddit_name} - Music: https://www.davidcuttermusic.com / @dcuttermusic")

def compile_tags():
    return [subreddit_name, 'reddit', 'memes', channel_name, 'youtube']

def get_memes(optional_quantity = None, after_n = None):
    reddit = praw.Reddit(client_id="Gu4GLLdzPqcO_g",
                            client_secret="YzCGWtA5PL9HwjAnAULBgBF5-FSgTg",
                            password="daRUdr107@",
                            user_agent="reddit_memebot",
                            username="TheRussianDraco")

    if optional_quantity == "None":
        optional_quantity = None
    if after_n == "None":
        after_n = None

    if optional_quantity == None:
        quantity = input("How many memes would you like to take?")
    else:
        quantity = optional_quantity
    
    if after_n == None:
        after_n = 0

    if quantity == 0 or quantity == '' or not quantity.isnumeric() or quantity == '0':
        quantity = 10

    subreddit = reddit.subreddit(subreddit_name) #The error message here

    all_submissions = []

    top = subreddit.hot(limit = int(after_n) + int(quantity)) #Can either be top(top of all time), hot(hots posts rn) or new(newest posts)

    n = 0

    for sub in top:
        if n > after_n:
            all_submissions.append(sub)
        n += 1

    print("Memes recieved")

    return [all_submissions, quantity]

def get_all_titles(all_subs):
    all_titless = []
    n = 0
    for sub in all_subs:
        if sub.title == "" or sub.title == " " or sub.title == "\n": 
            print("EMPTY TITLE")
            all_titless.append(f"Empty title number {str(n)}")
            n += 1
        else:
            print(sub.title)
            additon = str(sub.title)
            all_titless.append(additon)

    print("Titles collected")

    return all_titless

def create_vidfolder(video_title, all_titles):
    #seperater = ''

    #seperater_options = ['~', '`', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '-', '=', '+', '[', ']', '{', '}', '|', ':', ':', '>', '<', '/', '?', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

    #for option in seperater_options:
    #    if not option in all_titles:
    #        seperater = option
    #        break

    #all_titles = str(seperater.join(all_titles)).split(seperater)

    folderpath = origin + video_title
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)

    folder_path = origin + video_title
    txt_name = video_title + ' titles.txt'
    txtpath = origin + video_title + '/' + txt_name
    log_path = txtpath

    #test_txt = origin + video_title + '/test.txt'
    #all_chars = list(''.join(all_titles).replace('\n', seperater))

    #if not os.path.exists(test_txt):
    #    with open(test_txt, 'x') as tstxt:
    #        tstxt.write('.')
    #    tstxt.close()

    #n = 0
    #invalid_chars = []

    #for char in all_chars:
    #    try:
    #        with open(test_txt, 'w') as tstxt:
    #            tstxt.write(char)
    #        tstxt.close()
    #    except UnicodeEncodeError:
    #        invalid_chars.append(str(n))
    #        print("Invalid title")
    #    n += 1

    #if len(invalid_chars) > 0:
    #    n = 0
    #    new_chars = []
    #
    #    for char in invalid_chars:
    #        if not str(n) in invalid_chars:
    #            new_chars.append(char)
#
    #    if os.path.exists(test_txt):
    #        os.remove(test_txt)
#
    #    all_titles = (''.join(new_chars)).replace(seperater, '\n')
#
    #    with open(txtpath, 'x') as txtfile:
    #        txtfile.write(str(''.join(new_chars)).replace(seperater, '\n'))
    #    txtfile.close()
    #else:
    #    for titl in all_titles:
    #        with open(txtpath, 'w') as txt:
    #            txt.write(titl)
    #        txt.close()
#
    #    os.remove(test_txt)

    final_title_array = []

    for title in all_titles:
        final_title_array.append(clean_title(title))

    alltitles = '\n'.join(final_title_array)

    with open(txtpath, 'w') as file:
        file.write(alltitles)
    file.close()
        

    print("Folder created")

    return [folder_path, txtpath]

def download_images(all_subs, log_path):
    n = 0
    error_subs = []

    for sub in all_subs:
        src = sub.url
        try:
            img = urlopen(src).read()
            open(r'C:/Users/mathn/' + 'raw' + str(n) + img_filetype, "wb").write(img) #'/'.join(log_path.split('/').pop(len(log_path) - 1)) + '/' + str(n) + img_filetype

        except HTTPError:
            error_subs.append(n)
            n -= 1
        except URLError:
            error_subs.append(n)
            n -= 1

        try:
            opimg = Image.open(r'C:/Users/mathn/' + 'raw' + str(n) + img_filetype)
        except PIL.UnidentifiedImageError:
            os.remove(r'C:/Users/mathn/' + 'raw' + str(n) + img_filetype)
            error_subs.append(n)
            n -= 1

        n += 1

    if len(error_subs) > 0:
        for x in error_subs:
            replace_line(log_path, int(x), '')

    print("Images downloaded")

    return n

def add_black_square(image_number):
    n = 0

    for img in glob.glob(r'C:/Users/mathn/*.jpg'):
        merge_images(black_rect, r'C:/Users/mathn/raw' + str(n) + img_filetype).save(r'C:/Users/mathn/' + 'rect' + str(n) + img_filetype)

        n += 1

        if n > image_number or n == image_number:
            return

def merge_images(file1, file2): #Always put black_rect first
    black_rect1 = Image.open(file1)
    try:
        image2 = Image.open(file2)
    except PIL.UnidentifiedImageError:
        print("PIL Image Error")
        
    (width2, height2) = image2.size

    base_width = image2.size[0]
    width_percent = (base_width / float(black_rect1.size[0]))
    hsize = int((float(black_rect1.size[1]) * float(width_percent)))
    black_rect1 = black_rect1.resize((base_width, hsize), PIL.Image.ANTIALIAS)
    black_rect1.save('C:/Users/mathn/black_rect.jpg')
 
    image1 = Image.open(file1)

    (width1, height1) = image1.size

    width1 = width2


    result_width = width1

    result_height = height1 + height2
 
    result = Image.new('RGB', (result_width, result_height))
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(0, height1))

    return result

def add_titles(txtpath, image_number, font_size=9, stroke_width=5):
    n = 0
    for img in glob.glob(r'C:/Users/mathn/*.jpg'):
        with open(txtpath, "r") as txtfile:
            txt_array = txtfile.read().split('\n')
        txtfile.close()

        top_text = txt_array[n]

        im = Image.open(r'C:/Users/mathn/rect' + str(n) + img_filetype)
        d1 = ImageDraw.Draw(im)
        myFont = ImageFont.truetype(r'C:/Users/mathn/OneDrive/Desktop/YoutubeMemePyProject/arial.ttf', 40)
        d1.text((28, 36), top_text, fill=(255, 255, 255), font = myFont)
        im.save(r'C:/Users/mathn/' + str(n) + img_filetype)


        n += 1

        if n > image_number or n == image_number:
            print("Titles added")
            return

def delete_raw_images(image_number):
    n = 0

    if os.path.exists(r'C:/Users/mathn/black_rect.jpg'):
        os.remove(r'C:/Users/mathn/black_rect.jpg')

    for img in glob.glob(r'C:/Users/mathn/*.jpg'):
        os.remove(r'C:/Users/mathn/rect' + str(n) + img_filetype)
        os.remove(r'C:/Users/mathn/raw' + str(n) + img_filetype)

        n += 1

        if n > image_number or n == image_number:
            return

def resize_images(image_number):
    resize_rect = Image.open(origin + 'resize_rect.jpg')

    req_size = []

    topwidth = 0
    topheight = 0

    for im in glob.glob(r'C:/Users/mathn/*.jpg'):
        img = Image.open(r'C:/Users/mathn/' + im[len(list(r'C:/Users/mathn/')):])

        trywidth, tryheight = img.size

        if trywidth > topwidth:
            topwidth, emptyheight = img.size

        if tryheight > topheight:
            emptywidth, topheight = img.size

        req_size.append(topwidth)
        req_size.append(topheight)

    sqr_array = []

    n = 0

    meme_n = 0

    for im in glob.glob(r'C:/Users/mathn/*.jpg'):
        char_array = list(im)
        point_loc = char_array.index('.')
        if char_array[point_loc - 1].isnumeric():
            meme_n += 1

    for im in glob.glob(r'C:/Users/mathn/*.jpg'):
        img = Image.open(im).convert("RGBA")

        w, h = img.size

        if w > h:
            larger_length = w
        elif h < w:
            larger_length = h
        else:
            larger_length = (h + w) // 2

        resize_rect.resize((larger_length, larger_length), Image.ANTIALIAS).save(r'C:/Users/mathn/resize_rect.png')

        resize_rect1 = Image.open(r'C:/Users/mathn/resize_rect.png').convert("RGBA")

        x2, y2 = resize_rect1.size

        sqr_array.append(larger_length)

        x = (x2 - w)//2
        y = 0

        if x2 < w or y2 < h:
            if w > h:
                larger_length = w
            if h > w:
                larger_length = h
            if h == w:
                larger_length = h

            resize_rect1.resize((larger_length, larger_length), Image.ANTIALIAS).save(r'C:/Users/mathn/resize_rectX.png')
            os.remove(r'C:/Users/mathn/resize_rect.png')
            os.rename(r'C:/Users/mathn/resize_rectX.png', r'C:/Users/mathn/resize_rect.png')
            resize_rect1 = Image.open(r'C:/Users/mathn/resize_rect.png').convert("RGBA")

            x2, y2 = resize_rect1.size
            if x2 < w or y2 < h:
                print("WARNING, the resize rectangle is STILL smaller than the meme, memes will be cut off")

        resize_rect1.paste(img, (x, y))
        resize_rect1.save(r'C:/Users/mathn/final' + im[len(list(r'C:/Users/mathn/')):].split('.')[0] + '.png', format = 'PNG')

        os.remove(r'C:/Users/mathn/' + im[len(list(r'C:/Users/mathn/')):])
        os.remove(r'C:/Users/mathn/resize_rect.png')

        n += 1

    sqr_sum = 0

    for x in sqr_array:
        sqr_sum += x

    sqr_mean = sqr_sum//len(sqr_array)

    n = 0

    for im in glob.glob(r'C:/Users/mathn/*.png'):
        x1, y1 = Image.open(im).size
        img = Image.open(r'C:/Users/mathn/' + im[len(list(r'C:/Users/mathn/')):])
        ###Add margins at the bottom and top###------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        img.resize((sqr_mean, sqr_mean), PIL.Image.ANTIALIAS).save((r'C:/Users/mathn/final' + im[len(list(r'C:/Users/mathn/')):]))
        os.remove((r'C:/Users/mathn/final' + im[len(list(r'C:/Users/mathn/')):]).replace('finalfinal', 'final'))
        os.rename(((r'C:/Users/mathn/final' + im[len(list(r'C:/Users/mathn/')):])), ((r'C:/Users/mathn/final' + im[len(list(r'C:/Users/mathn/')):]).replace('finalfinal', 'final')))
        x1, y1 = Image.open((r'C:/Users/mathn/final' + im[len(list(r'C:/Users/mathn/')):]).replace('finalfinal', 'final')).size
        n += 1

    print("Images resized")

def get_thumbnail(image_number, folder_path):
    random_meme = random.randint(0, image_number - 1)

    n = 0

    for img in glob.glob(r'C:/Users/mathn/*.png'):
        n += 1

        if n == random_meme:
            Image.open(img).save(folder_path + '/thumbnail.png')
            thumbnail_path = folder_path + '/thumbnail.png'

def make_vid(image_number, video_name, lengths):
    #Deletes wrong images:
    for img in glob.glob(r'C:/Users/mathn/*.png'):
        if not 'final' in img:
            os.remove(r'C:/Users/mathn/' + img[len(list(r'C:/Users/mathn/')):])

    xs = []
    ys = []

    for im in glob.glob(r'C:/Users/mathn/*.png'):
        if 'final' in im:
            imo = Image.open(im)
            h, w = imo.size
            ys.append(h)
            xs.append(w)

    for val in xs:
        for val2 in xs:
            if not val == val2:
                print(str(val) + " - :vs: - " + str(val2))

    for val in ys:
        for val2 in ys:
            if not val == val2:
                print(str(val) + " - :vs: - " + str(val2))



    image_folder = r'C:/Users/mathn'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    n = 0

    video_n_array = []

    for image in images:
        frame_rate = 1 / lengths[n]
        print(f'F_R: {str(frame_length)}')
        video = cv2.VideoWriter(r'C:/Users/mathn/' + str(n) + '.avi', 0, frame_rate, (width,height)) #Name, fourcc?, frame_rate, vid_width_height - #Lengths should be in seconds
        video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()

        subprocess.call(['ffmpeg', '-i', r'C:/Users/mathn/' + str(n) + '.avi', r'C:/Users/mathn/' + str(n) + '.mp4'])
        if os.path.exists(r'C:/Users/mathn/' + str(n) + '.avi'):
            os.remove(r'C:/Users/mathn/' + str(n) + '.avi')

        if os.path.exists(r'C:/Users/mathn/' + str(n) + '.mp4'):
            video_n_array.append(r'C:/Users/mathn/' + str(n) + '.mp4')
        else:
            print("mp4 memes section dosent exist")

        n += 1

    video_array = []

    for vid in video_n_array:
        video_array.append(VideoFileClip(vid))

    final_vid = concatenate_videoclips(video_array)

    final_vid.write_videofile(video_name.split('.')[0] + '.mp4')

    for vd in glob.glob(r'C:/Users/mathn/*.mp4'):
        if not video_name.split('/')[-1].split('.')[0] in vd:
            os.remove(vd)

    subprocess.call(['ffmpeg', '-i', video_name.split('.')[0] + '.mp4', video_name])

    os.remove(video_name.split('.')[0] + '.mp4')

    #conv = Converter()

    #info = conv.probe(video_name.split('.')[0] + '.mp4')

    #convert = conv.convert()

    print("Video created")

def delete_allimages():
    for img in glob.glob(r'C:/Users/mathn/*.png'):
        os.remove(img)

def grab_music(video_path):
    video = mpe.VideoFileClip(video_path)
    vid_duration = round(int(video.duration))

    music_array = []

    for music in glob.glob(music_folder + '/*.mp3'):
        music_info = music[len(list(r'C:/Users/mathn/')):]
        music_path = music_folder + '/' + music_info
        array_addition = [music_info, str(round(MP3(music_path).info.length))]
        music_array.append(array_addition)

    for song in music_array:
        music_choice = random.randint(0, len(music_array) - 1)
        if int(music_array[music_choice][1]) > int(vid_duration):
            raw_name = song[0]
            s2_path = r'C:/Users/mathn/' + raw_name
            if not r'C:/Users/mathn/' in s2_path:
                s2_path = r'C:/Users/mathn/' + s2_path

            ffmpeg_extract_subclip(s2_path, 0, vid_duration, targetname = r'C:/Users/mathn/tempfinalmusic.mp3')

            os.rename(r'C:/Users/mathn/tempfinalmusic.mp3', r'C:/Users/mathn/finalmusic.mp3')

            return

    final_music_len = 0
    final_musics = []

    while final_music_len < vid_duration:
        music_choice = random.randint(0, len(music_array) - 1)
        if len(music_array) < 6:
            music_choice.pop(music_choice)
        final_music_len += music_array[music_choice][1]
        final_musics.append(music_array[music_choice])
        if final_music_len > vid_duration:
            break

    n = 0

    rawfinal_music = None

    AudioSegment.converter = r"C:\Users\mathn\OneDrive\Desktop\YoutubeMemePyProject\ffmpeg.exe"
    AudioSegment.ffprobe = r"C:\Users\mathn\OneDrive\Desktop\YoutubeMemePyProject\ffprobe.exe"

    for song in final_musics:
        if n == 0:
            rawfinal_music = AudioSegment.from_mp3(Path(music_folder + '/' + str(song.filename)[len(list(music_folder)):]))
        raw_song = AudioSegment.from_mp3(Path(music_folder + '/' + str(song.filename)[len(list(music_folder)):]))

        rawfinal_music = rawfinal_music + AudioSegment.silent(duration=5000) + raw_song

        n = 1

    final_song = rawfinal_music[0:vid_duration * 1000]
    final_song.export(r'C:/Users/mathn/finalmusic.wav', format = "wav")

    print("Music created")

def read_titles(txtpath):
    with open(txtpath, "r") as txtfile:
        txt_array = txtfile.read().split('\n')
    txtfile.close()

    engine = pyttsx3.init()

    voices = engine.getProperty('voices')
    rate = engine.getProperty('rate')
    engine.setProperty('voice', voices[0].id)
    engine.setProperty('rate', 115)
    engine.setProperty('ages', 1) #If error change to 'age'

    n = 0

    if len(txt_array) < 1:
        print("TXT File is empty")

    title_n = len(txt_array)

    for title in txt_array:
        if title == ' ' or title == None or title == '\n':
            print(f"Empty title number {str(n)}")
        else:
            engine.save_to_file(text = str(title).replace('*', ''), filename = r'C:/Users/mathn/' + str(txt_array.index(title)) + '.wav')
            engine.runAndWait()
            AudioSegment.from_wav(Path(r'C:/Users/mathn/' + str(txt_array.index(title)) + '.wav')).export(r'C:/Users/mathn/' + str(txt_array.index(title)) + '.mp3', format = 'mp3')
            os.remove(r'C:/Users/mathn/' + str(txt_array.index(title)) + '.wav')
        
        n += 1

    if n < 1:
        print("No titles were made")

    n = 0
    nn = 0
    final_title_compliation = None
    music_compile = []

    for music in glob.glob(music_folder + '/*.mp3'):
        music_compile.append(music.split('.')[1].replace('.mp3', ''))

    voice_array = []

    for vc in glob.glob(r'C:/Users/mathn/*.mp3'):
        for m in music_compile:
            if not m in vc:
                print(vc)
                voice_array.append(vc)

    n_t = 0

    for x in range(0, title_n):
        voice_array.append(r'C:/Users/mathn/' + str(n_t) + '.mp3')
        n_t += 1

    last_vc = ""

    if len(voice_array) < 1:
        print("WARNING - No voices have been created")
    else:
        last_vcn = -1
        last_vc = voice_array[last_vcn]
        if not os.path.exists(last_vc):
            while not os.path.exists(last_vc):
                last_vcn -= 1
                last_vc = voice_array[last_vcn]
                if os.path.exists(last_vc):
                    break
        print(f'LAST_VC: {last_vc}')

    for voice in voice_array:
        if not os.path.exists(voice):
            break

        v1_path = voice
        #max_time = 6*1000

        #audio = audioread.audio_open(v1_path)
        audio = MP3(v1_path)

        audio_length = int(audio.info.length)

        about_pause = 4000 #MilliSeconds

        voice_margin = 3500#max_time - audio.info.length #MP3(v1_path).info.length

        AudioSegment.converter = r"C:\Users\mathn\OneDrive\Desktop\YoutubeMemePyProject\ffmpeg.exe"
        AudioSegment.ffprobe = r"C:\Users\mathn\OneDrive\Desktop\YoutubeMemePyProject\ffprobe.exe"

        import_path = Path(v1_path)

        #Calculate the voice_margin to make title + silence equal to a natural number
        round_len = int(round(audio_length))

        voice_margin = ((audio_length + about_pause / 1000) - round_len)

        #if audio_length + voice_margin == round(audio_length + voice_margin):
        #    print('Voice margin - Outcome = TRUE')
        #else:
        #    print('Voice margin - Outcome = FALSE')
        #    print(str(audio_length) + ' + ' + str(voice_margin) + ' - ' + str(audio_length + voice_margin) +' != ' + str(round(audio_length + voice_margin)))

        total = audio_length + voice_margin

        frame_length.append(total)

        temp_voice = AudioSegment.from_mp3(import_path)
        #temp_voice = get_mp3f(v1_path)
        temp_silence = AudioSegment.silent(duration = voice_margin)

        final_voice = temp_voice + temp_silence #Make a join wav file function instead of this
        if n == 0:
            final_title_compliation = final_voice
        else:
            final_title_compliation = final_title_compliation + final_voice
        n = 1
        nn += 1

        if voice == last_vc:
            final_title_compliation.export(r'C:/Users/mathn/finaltitle.wav', format = 'wav')
            for vc2 in voice_array: #Deletes seperate titles
                if os.path.exists(vc2):
                    os.remove(vc2)

    if len(frame_length) == len(voice_array):
        print("All frame lengths recieved correctly")

    print("Titles read")

    #final_title_compliation.export(r'C:/Users/mathn/finaltitle.wav', format = "wav") #Relocated to into the loop bc of Nonetype problems

def mix_narration_music():
    #final_titles = AudioSegment.from_wav(r'C:/Users/mathn/finaltitle.wav')
    #final_musics = AudioSegment.from_wav(r'C:/Users/mathn/finalmusic.wav')

    AudioSegment.converter = r"C:\Users\mathn\OneDrive\Desktop\YoutubeMemePyProject\ffmpeg.exe"
    AudioSegment.ffprobe = r"C:\Users\mathn\OneDrive\Desktop\YoutubeMemePyProject\ffprobe.exe"

    AudioSegment.from_mp3(r"C:/Users/mathn/finalmusic.mp3").export(r"C:/Users/mathn/finalmusic.wav", format = 'wav')

    os.remove(r"C:/Users/mathn/finalmusic.mp3")

    final_titles = AudioSegment.from_wav(Path(r"C:/Users/mathn/finaltitle.wav"))
    final_musics = AudioSegment.from_wav(Path(r"C:/Users/mathn/finalmusic.wav"))

    quiet_music = final_musics - 30

    final_sounds = final_titles.overlay(quiet_music, position = 0)

    final_sounds.export(r'C:/Users/mathn/finalsound.mp3', format = "mp3")

    os.remove(r'C:/Users/mathn/finaltitle.wav')
    os.remove(r'C:/Users/mathn/finalmusic.wav')

    print("Audio mixed")

def overlay_sound(vid_title):
    video = mpe.VideoFileClip(r'C:/Users/mathn/video' + vid_filetype)
    sound = mpe.AudioFileClip(r'C:/Users/mathn/finalsound.mp3')

    final = video.set_audio(sound)
    final.write_videofile(r'C:/Users/mathn/' + vid_title + vid_filetype, codec= 'mpeg4', audio_codec = 'libvorbis')

    #os.remove(r'C:/Users/mathn/video' + vid_filetype)

    print("Audio overlayed")

def post_vid(title, desc, tags, status, video_path):
    print("Starting video posting")

    keywords = ','.join(tags)

    os.system(f'python upload_video.py --file="{video_path}" --title="{title}" --description="{desc}" --keywords="{keywords}" --category="24" --privacyStatus="{status}"')

def save_video_name(video_name):
    with open(r'C:/Users/mathn/OneDrive/Desktop/YoutubeMemePyProject/vid_logbook.txt', 'a') as logbook:
        logbook.write('\n' + video_name)
    logbook.close()

    print("Video name logged")

def remove_video(vid_title):
    if post_option == 'y':
        os.remove(r'C:/Users/mathn/' + vid_title + vid_filetype)

def remove_thumbnail(thumbnail):
    if os.path.exists(thumbnail):
        os.remove(thumbnail)
    else:
        print("WARNING - Thumbnail was never created, cannot be deleted")


def main():
    vidtitle = generate_title()
    raw_memereturn = get_memes()
    all_subss = raw_memereturn[0]
    quantity = raw_memereturn[1]
    all_titlesss = get_all_titles(all_subss)
    path_array = create_vidfolder(vidtitle, all_titlesss)
    folder_path = path_array[0]
    txt_path = path_array[1]
    img_number = download_images(all_subss, log_path)
    add_black_square(img_number)
    add_titles(txtpath = txt_path, image_number = img_number)
    delete_raw_images(img_number)
    resize_images(img_number)
    get_thumbnail(img_number, folder_path)
    read_titles(txt_path)
    make_vid(img_number, r'C:/Users/mathn/' + 'video' + vid_filetype, frame_length)
    delete_allimages()
    grab_music(r'C:/Users/mathn/video' + vid_filetype)
    mix_narration_music()
    overlay_sound(vidtitle)
    if post_option == 'y':
        post_vid(vidtitle, compile_description(), compile_tags(), video_status, r'C:/Users/mathn/' + vidtitle + vid_filetype)#, thumbnail_path)
        print("Video Posted!")
    save_video_name(vidtitle)
    remove_video(vidtitle)
    remove_thumbnail(thumbnail_path)

    print("All steps complete - Video created")
    #if os.path.exists(r'C:/Users/mathn/finalsound.mp3'):
        #os.remove(r'C:/Users/mathn/finalsound.mp3')
    #if os.path.exists(r'C:/Users/mathn/video' + vid_filetype):
        #os.remove(r'C:/Users/mathn/video' + vid_filetype)

main()
print("video and audio doesnt connect")