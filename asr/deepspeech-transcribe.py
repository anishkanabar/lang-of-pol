# NOTE: To use this must have virtual env set:
#  virtualenv -p python3 $HOME/tmp/deepspeech-venv/
#  source $HOME/tmp/deepspeech-venv/bin/activate
# 
#

import glob
import subprocess
import os
import pandas as pd
import wave
import numpy as np
from deepspeech import Model
from timeit import default_timer as timer

def check_paths(trans_path, feed, date, mp3_file):
	#Ensure file paths exist
	try:
		zone_path = trans_path+'/'+feed.replace(' ','')+'/'
		if not os.path.isdir(zone_path):
			os.mkdir(zone_path)
		date_path = zone_path+'/'+date+'/'
		if not os.path.isdir(date_path):
			os.mkdir(date_path)
		file_path = date_path+'/'+mp3_file.split('.')[0]+'/'
		if not os.path.isdir(file_path):
			os.mkdir(file_path)
		return zone_path, date_path, file_path
	except:
		print('Problem checking/creating paths for %s %s %s' % (feed, date, mp3_file))

def transcribe_mp3(feed, date, mp3_file, alpha, beta, beam, gt=False, gt_file='', thresh='-24db', trans_path='/media/2tbcrypt/data'):
	#Get paths
	trans_path='/media/2tbcrypt/data'
	data_path=trans_path.replace('2tbcrypt','4tb')
	_, _, file_path = check_paths(trans_path, feed, date, mp3_file)
	#Get list of files to process
	#wavs_path = data_path+'/'+feed.replace(' ','')+'/'+date+'/'+mp3_file.split('.')[0]+'/'
	wavs_to_transcribe = glob.glob(file_path.replace('2tbcrypt','4tb')+'*'+thresh+'.wav')
	wavs_to_transcribe.sort()
	#Load model
	ds = Model(aModelPath='/home/chris/deepspeech/DeepSpeech/models/output_graph.pbmm', aBeamWidth=beam)
	ds.enableDecoderWithLM(aLMPath='/home/chris/deepspeech/DeepSpeech/models/lm.binary', 
		aTriePath='/home/chris/deepspeech/DeepSpeech/models/trie', 
		aLMAlpha=alpha, 
		aLMBeta=beta)
	#Process files
	total_audio_length = 0
	inference_start = timer()
	for wav in wavs_to_transcribe:
		file_name = wav.replace('.wav','.txt').replace('4tb','2tbcrypt')
		fin = wave.open(wav, 'rb')
		fs = fin.getframerate()
		audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)
		audio_length = fin.getnframes() * (1/fs)
		total_audio_length += audio_length
		fin.close()
		transcription = ds.stt(audio)
		with open(file_name,'w') as transcription_file:
			transcription_file.write(transcription+'\n')
		'''
		command = ' '.join(['deepspeech',
			'--model /home/chris/deepspeech/DeepSpeech/models/output_graph.pbmm',
			'--lm /home/chris/deepspeech/DeepSpeech/models/lm.binary',
			'--trie /home/chris/deepspeech/DeepSpeech/models/trie',
			'--audio %s' % (wav),
			'--lm_alpha %s' % (alpha),
			'--lm_beta %s' % (beta),
			'--beam_width %s' % (beam),
			'> %s' % (file_name)])
		subprocess.run(command, shell=True)
		'''
	inference_end = timer() - inference_start
	print('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, total_audio_length))
	#Combine into one transcript
	with open(file_path+'transcribe'+thresh+'_a'+str(alpha)+'_b'+str(beta)+'_beam'+str(beam)+'.txt','w') as transcription:
		if gt:
			line_names = []
			with open(gt_file,'r') as f:
				lines = f.readlines()
				for line in lines:
					data=line.split(' ')
					line_names.append(data[0])
			n=0
			for wav in wavs_to_transcribe:
				file_name = wav.replace('.wav','.txt').replace('4tb','2tbcrypt')
				with open(file_name) as infile:
					for line in infile:
						transcription.write(line_names[n].strip()+' '+line.upper())
				n+=1
		else:
			n=0
			for wav in wavs_to_transcribe:
				file_name = wav.replace('.wav','.txt').replace('4tb','2tbcrypt')
				with open(file_name) as infile:
					for line in infile:
						transcription.write('Utt '+str(n)+': '+line.upper())
				n+=1
	print("Done trascribing %s with alpha %s beta %s beam_width %s" % (mp3_file, str(alpha), str(beta), str(beam)))

mp3_file='201808042331-339616-27730.mp3'
feed='Zone 1'
date='2018_08_05'
gt_file='/home/chris/Documents/segments'

#alpha=0.30 # float
#beta=-1 # float
#beam=500 # int
#file = '/media/2tbcrypt/data/Zone1/2018_08_05/201808042331-339616-27730/transcribe-0db_a' + str(alpha) + '_b' + str(beta) + '_beam' + str(beam) + '.txt'
#transcribe_mp3(feed, date, mp3_file, alpha, beta, beam, gt=True, gt_file=gt_file, thresh='-0db')

#transcribe_mp3(feed, date, mp3_file, alpha, beta, beam, gt=True, gt_file=gt_file, thresh='-0db')


alpha_range = [i/100 for i in range(25,36,1)]
beta_range = [-i/10 for i in range(8,16,1)]
beam_range = [15000,25000] 
len(alpha_range)*len(beta_range)*len(beam_range)*7/60

for alpha in alpha_range:
	for beta in beta_range:
		for beam in beam_range:
			file = '/media/2tbcrypt/data/Zone1/2018_08_05/201808042331-339616-27730/transcribe-0db_a' + str(alpha) + '_b' + str(beta) + '_beam' + str(beam) + '.txt'
			if not os.path.exists(file):
				transcribe_mp3(feed, date, mp3_file, alpha, beta, beam, gt=True, gt_file=gt_file, thresh='-0db')

# Wrapper to call Kaldi's compute-wer from command line and parse %WER
def get_wer(transcription_file, reference_file):
	command = ' '.join(['compute-wer',
		'--mode=strict',
		'ark:"%s"' % (reference_file),
		'ark,p:"%s"' % (transcription_file)])
	proc = subprocess.run(command, shell=True, capture_output=True)
	wer = float(proc.stdout.decode('utf-8').split(' ')[1])
	return wer

# Process a directory of transcriptions for %WER (TODO: Generate Pandas dataframe capturing parameter info from filename)
def get_wer_dir(transcription_dir, reference_file):
	wers_list=[]
	transcription_files = glob.glob(transcription_dir+'transcribe-0db*.txt')
	for file in transcription_files:
		only_filename = file.split('/')[-1].replace('.txt','')
		file_info = only_filename.split('_')[1:]
		alpha = float(file_info[0].replace('a',''))
		beta = float(file_info[1].replace('b',''))
		beam = int(file_info[2].replace('beam',''))
		wers_list.append([alpha, beta, beam, get_wer(file, reference_file)])
	df_wer = pd.DataFrame(wers_list, columns=['alpha','beta','beam','wer'])
	return df_wer

reference_file = '/home/chris/Documents/text'
transcription_dir = '/media/2tbcrypt/data/Zone1/2018_08_05/201808042331-339616-27730/'

df_wer = get_wer_dir(transcription_dir, reference_file)
df_wer['wer'].min()

df_wer.groupby('alpha')['wer'].min()

ax = df_wer.hist(column='wer', bins=25, grid=False, figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)
ax = ax[0]
for x in ax:
	# Despine
	x.spines['right'].set_visible(False)
	x.spines['top'].set_visible(False)
	x.spines['left'].set_visible(False)
	# Switch off ticks
	x.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
	# Draw horizontal axis lines
	vals = x.get_yticks()
	for tick in vals:
		x.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)
	# Remove title
	x.set_title("Parameter Tuning (\u03B1, \u03B2, beam width)", weight='bold')
	# Set x-axis label
	x.set_xlabel("WER%", labelpad=20, weight='bold', size=12)
	# Set y-axis label
	x.set_ylabel("Frequency", labelpad=20, weight='bold', size=12)

t = ax[0].get_figure()
t.savefig('/home/chris/Documents/wer_hist.png') 

df_wer_min = df_wer[df_wer['wer']==df_wer['wer'].min()]
if len(df_wer_min)>1:
	df_temp = df_wer_min.sample()
	alpha_3d = df_temp['alpha'].values[0]
	beta_3d = df_temp['beta'].values[0]
	beam_3d = df_temp['beam'].values[0]
else:
	df_temp = df_wer_min
	alpha_3d = df_temp['alpha'].values[0]
	beta_3d = df_temp['beta'].values[0]
	beam_3d = df_temp['beam'].values[0]

df_wer_alpha_min= df_wer[df_wer['alpha']==alpha_3d]
df_wer_beta_min= df_wer[df_wer['beta']==beta_3d]
df_wer_beam_min= df_wer[df_wer['beam']==beam_3d]


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D



fig = plt.figure()
ax = Axes3D(fig)

ax.set_title(r"Parameter Tuning ($\alpha$, $\beta$, beam width = 250)", weight='bold')
ax.set_xlabel(r"$\alpha$", labelpad=20, weight='bold', size=20)
ax.set_ylabel(r"$\beta$", labelpad=20, weight='bold', size=20)
ax.set_zlabel(r"$WER\%$", labelpad=20, weight='bold', size=20)
ax.plot_trisurf(df_wer_beam_min.alpha, df_wer_beam_min.beta, df_wer_beam_min.wer, cmap=cm.jet, linewidth=0.2)

plt.show()


Xalpha, Yalpha = np.meshgrid(df_wer_beam_min['alpha'].values, df_wer_beam_min['beta'].values)
Zalpha = df_wer_beam_min['wer'].values

surf = ax.plot_surface(Xalpha, Yalpha, Zalpha, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)

ax.set_zlim(-1.01, 1.01)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('Original Code')
plt.show()



