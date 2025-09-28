import { Button, Space } from 'antd';
import { useTranslation } from 'react-i18next';

const MicSVG = ({ vad, micOn }) => {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 384 512"
      fill={vad.userSpeaking && micOn ? '#B22222' : 'black'}
      width="22"
      height="22"
    >
      <path d="M192 0C139 0 96 43 96 96V256c0 53 43 96 96 96s96-43 96-96V96c0-53-43-96-96-96zM64 216c0-13.3-10.7-24-24-24s-24 10.7-24 24v40c0 89.1 66.2 162.7 152 174.4V464H120c-13.3 0-24 10.7-24 24s10.7 24 24 24h72 72c13.3 0 24-10.7 24-24s-10.7-24-24-24H216V430.4c85.8-11.7 152-85.3 152-174.4V216c0-13.3-10.7-24-24-24s-24 10.7-24 24v40c0 70.7-57.3 128-128 128s-128-57.3-128-128V216z" />
    </svg>
  );
};

const VoiceContinuationSVG = () => {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      fill="black"
      width="22"
      height="22"
      viewBox="0 0 24 24"
    >
      <g id="SVGRepo_bgCarrier" strokeWidth="0"></g>
      <g
        id="SVGRepo_tracerCarrier"
        strokeLinecap="round"
        strokeLinejoin="round"
      ></g>
      <g id="SVGRepo_iconCarrier">
        <g id="cycle">
          <g>
            <path d="M17.1,23.8L12.4,21l2.7-4.8l1.7,1l-1.7,3.1l3,1.8L17.1,23.8z M5.7,11L4,8L1.1,9.7L0.1,8l4.7-2.8L7.5,10L5.7,11z"></path>
          </g>
          <g>
            <polygon points="22,6 16.5,6 16.5,4 20,4 20,0.5 22,0.5 "></polygon>
          </g>
          <g>
            <path d="M15.4,21.5l-0.4-2c4-0.9,6.9-4.5,6.9-8.6c0-0.6-0.1-1.3-0.2-1.9l2-0.4c0.2,0.8,0.3,1.6,0.3,2.3 C24,16.1,20.4,20.5,15.4,21.5z"></path>
          </g>
          <g>
            <path d="M9.8,21.3C5.3,19.9,2.2,15.8,2.2,11c0-1.3,0.2-2.6,0.7-3.8l1.9,0.7c-0.4,1-0.6,2-0.6,3.1c0,3.9,2.5,7.2,6.1,8.4L9.8,21.3z "></path>
          </g>
          <g>
            <path d="M19.6,5c-1.7-1.9-4.1-3-6.6-3c-2.1,0-4.1,0.8-5.7,2.1L6,2.6C7.9,0.9,10.4,0,13,0c3.1,0,6,1.3,8.1,3.6L19.6,5z"></path>
          </g>
        </g>
      </g>
    </svg>
  );
};

interface IProps {
  micOn: boolean;

  setMicOn(v: boolean): void;

  micAvailable: boolean;

  setMicAvailable(v: boolean): void;

  STTIng: boolean;

  setSTTIng(v: boolean): void;

  voiceContinuationEnable: boolean;

  setVoiceContinuationEnable(v: boolean): void;

  setUserSpeakLately(v: any): void;

  timerRef: React.Ref<any>;

  setTimerInterval(v: any): void;

  vadTimeoutMS: number;
  vad: object;
  inputDisabled: boolean;
  messageInput: string;
  handlePressEnter(): void;
  voiceContinueTimerRef: React.Ref<any>;
}

export const STTButton = ({
  micOn,
  micAvailable,
  setMicAvailable,
  vad,
  setSTTIng,
  setMicOn,
  setVoiceContinuationEnable,
  voiceContinuationEnable,
  setUserSpeakLately,
  timerRef,
  setTimerInterval,
  vadTimeoutMS,
  inputDisabled,
  voiceContinueTimerRef,
  handlePressEnter,
  messageInput,
}: IProps) => {
  const { t } = useTranslation();
  const handleToggleMic = async (e: any) => {
    e.preventDefault();
    let _micAvailable = false;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach((track) => track.stop());
      _micAvailable = true;
    } catch (err) {
      setMicOn(false);
    } finally {
      setMicAvailable(_micAvailable);
    }

    try {
      if (_micAvailable) {
        setSTTIng(false);
        if (micOn) {
          vad.stop();
          setMicOn(false);
          setVoiceContinuationEnable(false);

          if (timerRef.current) {
            clearInterval(timerRef.current);
            clearInterval(voiceContinueTimerRef.current);
            setUserSpeakLately(false);
          }
        } else {
          setMicOn(true);
          vad.start();

          timerRef.current = setInterval(() => {
            setTimerInterval(true);
          }, vadTimeoutMS);
        }
      } else {
        console.error(t('micNotAvailToast'));
      }
    } catch (e) {
      console.log(e);
      console.error(t('micOpenErrToast'));
    }
  };

  return (
    <Space size={'small'} style={{ display: inputDisabled ? 'none' : '' }}>
      <Button
        type={'text'}
        onClick={handleToggleMic}
        style={{ backgroundColor: micOn ? '#86a6df' : 'beige' }}
        loading={vad?.loading}
        icon={<MicSVG vad={vad} micOn={micOn} />}
      ></Button>
      {micOn && (
        <Button
          style={{
            backgroundColor: voiceContinuationEnable ? '#86a6df' : 'beige',
          }}
          onClick={async (e) => {
            e.preventDefault();
            if (!micOn) {
              await handleToggleMic(e);
            }
            // alert("11"+voiceContinuationEnable);
            setVoiceContinuationEnable(!voiceContinuationEnable);
            if (voiceContinuationEnable) {
              if (timerRef.current) {
                clearInterval(voiceContinueTimerRef.current);
                setUserSpeakLately(false);
              }
            } else {
              if (messageInput.length > 0) {
                handlePressEnter();
              }
            }
          }}
          type={'text'}
          disabled={!micAvailable}
          loading={vad?.loading}
          icon={<VoiceContinuationSVG />}
        ></Button>
      )}
    </Space>
  );
};
