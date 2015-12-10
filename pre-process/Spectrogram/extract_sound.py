import os
import subprocess
import wave

# location --> ../../data/pre-process/Spectrogram/
#ffmpeg -i sound.avi -ab 160k -ac 1 -ar 44100 -vn sound_mono.wav -y

'''
Extract sound from the source video file into the output directory
'''
def extractWav(source, outdir):
    ''' If possible, extract the sound file for a given avi file
        
        Inputs:
            source: string, path to file
        Outputs:
            None
    '''

    output = (source.split('.')[-2]).split('/')[-1]
    output += '.wav';
    output = outdir +'/'+ output
    cmd = 'ffmpeg -i ' + source + ' -ab 160k -ac 1 -ar 44100 -vn ' \
          + output + ' -y'
    cmd = cmd.split(' ') 
    FNULL = open(os.devnull, 'w')
    #Note: I had to add shell=True to the parameters of the subprocess call
    #to get the script to run. 
    subprocess.call(cmd,stdout=FNULL, stderr=subprocess.STDOUT)

def main():
    dirs = [x[0] for x in os.walk("../../data/UCF-101/")]
    l = len(dirs)

    # make directories to store the files
    for i in xrange(l):
        if(i==0):
            continue
        
        category = dirs[i].split("/")[-1]
        if category == '':
            continue
        
        directory = "../../data/pre-process/Spectrogram/" + category
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # loop through all files 
    for i in xrange(l):
        print("%.2f"%(float(i)/float(l)))
        if(i==0):
            continue
        category = dirs[i].split("/")[-1]
        if category == '':
            continue
        outdir = "../../data/pre-process/Spectrogram/" + category
        
        for filename in os.listdir(dirs[i]):
            source = dirs[i]+'/'+filename
            extractWav(source, outdir)


if __name__ == '__main__':
    main()
    







