# # import cv2 as cv
# # from PIL import Image
# # import pytesseract as tess
# #
# # cap = cv.VideoCapture(0,cv.CAP_DSHOW)
# #
# # def recoginse_text(image):
# #
# #     # 灰度 二值化
# #     gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# #     # 如果是白底黑字 建议 _INV
# #     ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
# #
# #     # 形态学操作 (根据需要设置参数（1，2）)
# #     kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 2))
# #     morph1 = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
# #     kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 1))
# #     morph2 = cv.morphologyEx(morph1, cv.MORPH_OPEN, kernel)
# #
# #     # 黑底白字取非，变为白底黑字（便于pytesseract 识别）
# #     cv.bitwise_not(morph2, morph2)
# #     textImage = Image.fromarray(morph2)
# #
# #     text = tess.image_to_string(textImage,lang="eng+tha")
# #     print("%s" % text)
# #
# #
# # def main():
# #     tess.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
# #     # recoginse_text(img)
# #     # while (True):
# #     #     ret, frame = cap.read()
# #     #
# #     #     img = cv.resize(frame, (500, 500))
# #
# #     src = cv.imread("ko_2015-01-05_06-55-48.jpg")
# #     cv.imshow("src", src)
# #     recoginse_text(src)
# #     # if cv.waitKey(1) & 0xFF == ord('q'):
# #     #     break
# #
# #     cv.waitKey(0)
# #     cv.destroyAllWindows()
# #
# #
# # if __name__ == "__main__":
# #     main()


# from gtts import gTTS
# from pygame import mixer
# import tempfile ,time, threading
# import speech_recognition as sr
#
# r = sr.Recognizer()
#
#
# def speak(sentence, lang):
#     with tempfile.NamedTemporaryFile(delete=True) as fp:
#         tts=gTTS(text=sentence, lang=lang)
#         tts.save('{}.mp3'.format(fp.name))
#         mixer.init()
#         mixer.music.load('{}.mp3'.format(fp.name))
#         mixer.music.play(1)
#
# def Micro():
#     with sr.Microphone() as source:
#         print("Please wait. Calibrating microphone...")
#         r.adjust_for_ambient_noise(source, duration=2)
#         print("Say something!")
#         audio=r.listen(source)
#         return  audio
#
#
# # try:
# def speech():
#     print("Google Speech Recognition thinks you said:")
#     ab = r.recognize_google(Micro(), language="zh-TW")
#     # print(r.recognize_google(Micro(), language="zh-TW"))
#     if(ab == "語音助理你好"):
#         speak("你好請問要幹嘛", 'zh-tw')
#         time.sleep(2)
#         print("Google Speech Recognition thinks you said:")
#         ak = r.recognize_google(Micro(), language="zh-TW")
#         # print(r.recognize_google(Micro(), language="zh-TW"))
#         if(str(ak) == "你好"):
#             speak("我不好你要幹麻", 'zh-tw')
#             time.sleep(2)
#             rd = r.recognize_google(Micro(), language="zh-TW")
#             if(str(rd) == "沒事"):
#                 speak("無聊", 'zh-tw')
#                 time.sleep(2)
#         elif(str(ak) == "在幹嘛"):
#             speak("關你屁事", 'zh-tw')
#             time.sleep(2)
#         elif (str(ak) == "再見"):
#             speak("滾", 'zh-tw')
#             time.sleep(2)
#         elif(str(ak) == "打卡"):
#             speak("林柏鑫你好打卡成功", 'zh-tw')
#             time.sleep(3)
# speech()

# timer = threading.Timer(10, speech())
# timer.start()



# import numpy as np
# import numpy.fft as nf
# import matplotlib.pyplot as plt
# from pylab import *
# mpl.rcParams['font.sans-serif'] = ['SimHei']
# mpl.rcParams['axes.unicode_minus'] = False
#
# import scipy.io.wavfile as wf
#
# #讀取音訊檔案,將其按照取樣率離散化，返回取樣率和訊號
# #sample_reate:取樣率(每秒取樣個數),　sigs:每個取樣位移值。
# #原始音訊訊號,時域資訊
# sameple_rate,sigs = wf.read('音樂3.wav')
# print('取樣率:{}'.format(sameple_rate))
# print('訊號點數量:{}'.format(sigs.size))
#
# times = np.arange(len(sigs))/sameple_rate
# plt.figure('Filter',facecolor='lightgray')
#
# plt.title('時間域',fontsize=16)
# plt.ylabel('訊號',fontsize=12)
# plt.grid(linestyle=':')
# plt.plot(times[:200],sigs[:200],color='dodgerblue',label='噪聲訊號 ')#檢視前200個
# plt.legend()
# plt.show()
#
# #基於傅立葉變換，獲取音訊頻域資訊
# #繪製音訊頻域的: 頻域/能量影象
# freqs = nf.fftfreq(sigs.size, 1/sameple_rate)
# complex_arry = nf.fft(sigs)
# pows = np.abs(complex_arry)#取絕對值
#
# plt.title('頻率域',fontsize=16)
# plt.ylabel('能量',fontsize=12)
# plt.grid(linestyle=':')
# plt.semilogy(freqs,pows,color='green',label='噪聲 Freq')
# plt.legend()
# plt.show()
#
# fun_freq = freqs[pows.argmax()] #獲取頻率域中能量最高的#1000.0
# print('fun_freq',fun_freq)
# noised_idx = np.where(freqs != fun_freq)[0] #獲取所有噪聲的下標
# print('noised_idx',noised_idx)
# ca = complex_arry[:]#complex_arry為前面傅立葉變換得到的陣列
# ca[noised_idx] = 0 #高通濾波#將噪聲全部設為0
# filter_pows = np.abs(complex_arry)#過濾後的傅立葉變換資料，原始資料已被修改，用於化
#
# filter_sigs = nf.ifft(ca)#逆傅立葉變換
#
# plt.title('時域圖',fontsize=16)
# plt.ylabel('Signal',fontsize=12)
# plt.grid(linestyle=':')
# plt.plot(times[:200],filter_sigs[:200],color='red',label='降噪後的訊號')
# plt.legend()
# plt.show()
#
# plt.title('頻率域',fontsize=16)
# plt.ylabel('power',fontsize=12)
# plt.grid(linestyle=':')
# plt.plot(freqs,filter_pows,color='green',label='降噪後的頻譜能量圖')#filter_pows為降噪後的資料，尚未進行逆傅立葉變換
# plt.legend()
# plt.show()
#
# # filter_sigs = filter_sigs.astype('i2')#格式轉換
# wf.write('降噪.wav',sameple_rate,(filter_sigs * 2 ** 15).astype(np.int16))


import numpy as np
import numpy.fft as nf
import scipy.io.wavfile as wf
import matplotlib.pyplot as mp

# 读取音频文件
sample_rate, noised_signs = wf.read("音樂3.wav")
print(sample_rate, noised_signs.shape)  # 采样率 (每秒个数), 采样位移
noised_signs = noised_signs / (2 ** 15)
times = np.arange(noised_signs.size) / sample_rate  # x轴

# 绘制音频 时域图
mp.figure("Filter", facecolor="lightgray")
mp.subplot(221)
mp.title("Time Domain", fontsize=12)
mp.ylabel("Noised_signal", fontsize=12)
mp.grid(linestyle=":")
mp.plot(times[:200], noised_signs[:200], color="b", label="Noised")
mp.legend()
mp.tight_layout()

# 傅里叶变换 频域分析 音频数据
complex_ary = nf.fft(noised_signs)

fft_freqs = nf.fftfreq(noised_signs.size, times[1] - times[0])  # 频域序列
fft_pows = np.abs(complex_ary)     # 复数的摸-->能量  Y轴

# 绘制频域图
mp.subplot(222)
mp.title("Frequency", fontsize=12)
mp.ylabel("pow", fontsize=12)
mp.grid(linestyle=":")
mp.semilogy(fft_freqs[fft_freqs > 0], fft_pows[fft_freqs > 0], color="orangered", label="Noised")
mp.legend()
mp.tight_layout()

# 去除噪声
# fft_pows.argmax()
fund_freq = fft_freqs[1000]
noised_indices = np.where(fft_freqs != fund_freq)[0]
filter_fft = complex_ary.copy()
filter_fft = filter_fft[:]
filter_fft[noised_indices] = 0  # 噪声数据位置 =0
filter_pow = np.abs(filter_fft)

# 绘制去除噪声后的 频域图
mp.subplot(224)
mp.title("Filter Frequency ", fontsize=12)
mp.ylabel("pow", fontsize=12)
mp.grid(linestyle=":")
mp.plot(fft_freqs[fft_freqs > 0], filter_pow[fft_freqs > 0], color="orangered", label="Filter")
mp.legend()
mp.tight_layout()

# 对滤波后的数组，逆向傅里叶变换
filter_sign = nf.ifft(filter_pow).real

# 绘制去除噪声的 时域图像
mp.subplot(223)
mp.title("Filter Time Domain", fontsize=12)
mp.ylabel("filter_signal", fontsize=12)




mp.grid(linestyle=":")
mp.plot(times[:200], filter_sign[:200], color="b", label="Filter")
mp.legend()
mp.tight_layout()

# 重新写入新的音频文件
wf.write('filter.wav', sample_rate, (filter_sign * 2 ** 15).astype(np.int16))
mp.show()


# import numpy as np
# import numpy.fft as nf
# import matplotlib.pyplot as plt
# import scipy.io.wavfile as wf
#
# #读取音频文件,将其按照采样率离散化，返回采样率和信号
# #sample_reate:采样率(每秒采样个数),　sigs:每个采样位移值。
# #================1.原始音频信号,时域信息=================================
# sameple_rate,sigs = wf.read('音樂3.wav')
# print('采样率:{}'.format(sameple_rate))
# print('信号点数量:{}'.format(sigs.size))
# sigs = sigs/(2**15)
#
# times = np.arange(len(sigs))/sameple_rate
# plt.figure('Filter',facecolor='lightgray')
# plt.subplot(221)
# plt.title('Time Domain',fontsize=16)
# plt.ylabel('Signal',fontsize=12)
# plt.grid(linestyle=':')
# plt.plot(times[:178],sigs[:178],color='dodgerblue',label='Noised Signal')
# plt.legend()
#
# #==================2.转换为频率域信号===================================
# #基于傅里叶变换，获取音频频域信息
# #绘制音频频域的: 频域/能量图像
# freqs = nf.fftfreq(sigs.size, 1/sameple_rate)
# complex_arry = nf.fft(sigs)
# pows = np.abs(complex_arry)
# plt.subplot(222)
# plt.title('Frequence Domain',fontsize=16)
# plt.ylabel('power',fontsize=12)
# plt.grid(linestyle=':')
# plt.semilogy(freqs[freqs>0],pows[freqs>0],color='dodgerblue',label='Noised Freq')
# plt.legend()
#
# #==============第3步=================================================
# #将低能噪声去除后绘制音频频域的: 频率/能量图书
# fun_freq = freqs[pows.argmax()] #获取频率域中能量最高的
# noised_idx = np.where(freqs != fun_freq)[0] #获取所有噪声的下标
# ca = complex_arry[:]
# ca[noised_idx] = 0 #高通滤波
# filter_pows = np.abs(complex_arry)
#
# plt.subplot(224)
# plt.ylabel('power',fontsize=12)
# plt.grid(linestyle=':')
# plt.plot(freqs[freqs>0],filter_pows[freqs>0],color='dodgerblue',label='Filter Freq')
# plt.legend()
# #================第4步==============================================
# filter_sigs = nf.ifft(ca)
# plt.subplot(223)
# plt.title('Time Domain',fontsize=16)
# plt.ylabel('Signal',fontsize=12)
# plt.grid(linestyle=':')
# plt.plot(times[:178],filter_sigs[:178],color='dodgerblue',label='Filter Signal')
# plt.legend()
#
# #重新生成音频文件
# # filter_sigs = (filter_sigs*(2**15)).astype('i2')
# wf.write('filter.wav', sameple_rate, (filter_sigs * 2 ** 15).astype(np.int16))
#
# plt.tight_layout()
# plt.show()