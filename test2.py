import streamlit as st
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
import os
import tempfile
import threading
import json
import librosa
import soundfile as sf
from aip import AipSpeech
import speech_recognition as sr
import uuid  # 用于生成唯一的文件名

# 设置编码
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["LANG"] = "en_US.UTF-8"

# 百度语音识别配置（替换为你的API信息）
BAIDU_APP_ID = "119055690"
BAIDU_API_KEY = "Ui908c5FJoScXlNyK7zuXR1Z"
BAIDU_SECRET_KEY = "uqckQD7hKf90VgfjFnaclil4MZ8piEnE"
baidu_client = AipSpeech(BAIDU_APP_ID, BAIDU_API_KEY, BAIDU_SECRET_KEY)

# 选择API提供商
API_PROVIDERS = {
    "twapi": "https://twapi.openai-hk.com/v1",
    "nengyongai": "https://ai.nengyongai.cn",

}


def get_ai_response(user_prompt):
    try:
        # 创建LLM实例
        model = ChatOpenAI(
            model='gpt-4o-mini',
            api_key=st.session_state.get('API_KEY', ''),
            base_url=st.session_state.get('API_BASE_URL', API_PROVIDERS["nengyongai"])
        )

        # 创建对话链
        chain = ConversationChain(llm=model, memory=st.session_state['memory'])

        # 获取AI响应
        response = chain.invoke({'input': user_prompt})['response']
        return response

    except Exception as e:
        # 捕获并处理API调用异常
        error_msg = f"获取AI响应失败: {str(e)}"
        st.error(error_msg)
        return "抱歉，我暂时无法回答这个问题。请检查您的API密钥或稍后再试。"


def convert_mp3_to_wav(mp3_path, wav_path='temp.wav'):
    """使用纯Python方案将MP3转换为WAV格式"""
    try:
        y, sr = librosa.load(mp3_path, sr=16000)  # 加载并重采样到16kHz
        sf.write(wav_path, y, sr, subtype='PCM_16')  # 保存为16位PCM WAV
        return wav_path
    except Exception as e:
        print(f"Librosa转换失败: {str(e)}")
        return None


def speech_to_text(audio_path, is_mp3=False):
    wav_path = None
    try:
        if is_mp3:
            # 仅使用librosa方案（无需ffmpeg）
            wav_path = convert_mp3_to_wav(audio_path)
            if not wav_path:
                return "MP3转换失败，请检查librosa库"
            audio_file = wav_path
        else:
            audio_file = audio_path

        result = baidu_client.asr(
            get_file_content(audio_file),
            'wav', 16000,
            {'dev_pid': 1537}
        )

        if result.get('err_no') == 0:
            return ''.join(result.get('result', []))
        else:
            return f"识别失败: {result.get('err_msg', '未知错误')}"
    except Exception as e:
        return f"发生异常: {str(e)}"
    finally:
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)


def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


def handle_uploaded_file(file_input):
    if isinstance(file_input, str):  # 如果是文件路径
        file_path = file_input
        file_name = os.path.basename(file_path)
    else:  # 如果是 UploadedFile 对象
        # 创建临时文件保存上传的内容
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_input.name)[1]) as f:
            f.write(file_input.getbuffer())
            file_path = f.name
            file_name = file_input.name

    is_mp3 = file_name.lower().endswith('.mp3')
    text = speech_to_text(file_path, is_mp3)

    # 如果是临时文件，处理完后删除
    if not isinstance(file_input, str):
        os.remove(file_path)

    return text


def record_audio():
    """录制音频并保存为WAV文件"""
    r = sr.Recognizer()

    with sr.Microphone() as source:
        st.write(f"开始录音...")
        audio = r.listen(source)  # 录制音频

    # 生成唯一的文件名
    output_file = f"{uuid.uuid4()}.wav"

    # 保存为WAV文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio.get_wav_data())
        temp_wav = f.name

    # 确保音频格式符合百度API要求（16位单声道，16000Hz）
    try:
        # 使用pydub转换格式（需提前安装）
        from pydub import AudioSegment

        sound = AudioSegment.from_wav(temp_wav)
        # 转换为16位、16000Hz、单声道
        sound = sound.set_sample_width(2)  # 16位
        sound = sound.set_frame_rate(16000)
        sound = sound.set_channels(1)  # 单声道
        sound.export(output_file, format="wav")
        st.write(f"音频已保存至: {os.path.abspath(output_file)}")
    except ImportError:
        # 若未安装pydub，直接复制临时文件（可能格式不标准）
        import shutil
        shutil.copy(temp_wav, output_file)
        st.write(f"已保存原始WAV文件至: {os.path.abspath(output_file)}")
    finally:
        os.remove(temp_wav)

    # 更新 last_uploaded_file
    st.session_state['last_uploaded_file'] = output_file
    return output_file


# 页面设置
st.set_page_config(
    page_title="我的ChatGPT",
    page_icon="🤖",
    layout="wide"
)

st.title('我的ChatGPT')

with st.sidebar:
    st.header("配置")

    # API提供商选择
    api_provider = st.selectbox(
        "选择API提供商",
        list(API_PROVIDERS.keys()),
        index=0,
        key="api_provider_select"
    )

    st.session_state['API_BASE_URL'] = API_PROVIDERS[api_provider]

    # API密钥输入
    api_key_input = st.text_input(
        '请输入你的API Key：',
        type='password',
        key="api_key_input"
    )

    # 保存API密钥到session_state
    if api_key_input:
        st.session_state['API_KEY'] = api_key_input

    st.markdown("---")
    st.info("""
    **提示：**
    - 请确保输入的API密钥有效
    - 官方OpenAI API需要有效的OpenAI账户
    - 第三方服务可能有不同的使用限制
    """)

# 初始化会话状态
if 'messages' not in st.session_state:
    st.session_state['messages'] = [{'role': 'ai', 'content': '你好主人，我是你的AI助手，我叫小美'}]
    st.session_state['memory'] = ConversationBufferMemory(return_message=True)

if 'voice_text' not in st.session_state:
    st.session_state['voice_text'] = ""

if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ""

if 'submit_voice' not in st.session_state:
    st.session_state['submit_voice'] = False


# 显示历史消息
for message in st.session_state['messages']:
    role, content = message['role'], message['content']
    st.chat_message(role).write(content)

# 文件上传区域
uploaded_file = st.file_uploader("上传音频文件", type=['mp3', 'wav'])

# 清除旧的音频文件引用
def clear_old_audio_file():
    if 'last_uploaded_file' in st.session_state:
        if os.path.exists(st.session_state['last_uploaded_file']):
            os.remove(st.session_state['last_uploaded_file'])
        st.session_state.pop('last_uploaded_file', None)
# 处理上传的文件
if uploaded_file is not None:
    # 重置用户输入状态
    st.session_state['voice_text'] = ""
    st.session_state['user_input'] = ""
    st.session_state['submit_voice'] = False
    with st.spinner("正在处理音频文件..."):
        text = handle_uploaded_file(uploaded_file)
        st.session_state['voice_text'] = text
        st.session_state['user_input'] = text
        st.session_state['submit_voice'] = True
        st.session_state['last_uploaded_file'] = uploaded_file.name  # 记录当前文件名
        st.session_state.pop('uploaded_file', None)  # 清除session中的文件引用
        uploaded_file = None  # 清除当前文件引用

# 录制音频按钮
if st.button("录制音频"):
    clear_old_audio_file()  # 清除旧的音频文件引用

    # 重置用户输入状态
    st.session_state['voice_text'] = ""
    st.session_state['user_input'] = ""
    st.session_state['submit_voice'] = False
    st.session_state.pop('uploaded_file', None)

    with st.spinner("正在录制音频..."):
        output_file = record_audio()
        text = handle_uploaded_file(output_file)
        st.session_state['voice_text'] = text
        st.session_state['user_input'] = text
        st.session_state['submit_voice'] = True
        st.session_state['last_uploaded_file'] = output_file  # 记录当前文件名
        os.remove(output_file)  # 处理完后删除录音文件
# 输入区域
user_input = st.chat_input("输入文字或上传音频文件")
if user_input:
    # 清除音频相关状态
    st.session_state['voice_text'] = ""
    st.session_state['user_input'] = ""
    st.session_state['submit_voice'] = False
    st.session_state.pop('uploaded_file', None)

# 处理用户输入
if user_input or st.session_state.get('submit_voice', False):
    # 获取API密钥
    api_key = st.session_state.get('API_KEY', '')

    # 区分处理音频输入和文本输入
    if st.session_state.get('submit_voice', False):
        user_input = st.session_state['user_input']
        st.session_state['submit_voice'] = False
        input_type = "音频"
    else:
        input_type = "文本"

    # 检查API密钥
    if not api_key:
        st.warning('请在侧边栏输入您的API Key！')
        st.stop()

    # 显示用户消息
    st.chat_message('human').write(user_input)
    st.session_state['messages'].append({'role': 'human', 'content': user_input})

    # 记录输入类型（可选）
    st.session_state['last_input_type'] = input_type

    # 获取AI响应
    with st.spinner('AI正在思考，请等待……'):
        resp_from_ai = get_ai_response(user_input)
        st.chat_message('ai').write(resp_from_ai)
        st.session_state['messages'].append({'role': 'ai', 'content': resp_from_ai})

    # 处理完后重置音频输入状态
    if input_type == "音频":
        st.session_state['user_input'] = ""
        st.session_state['voice_text'] = ""