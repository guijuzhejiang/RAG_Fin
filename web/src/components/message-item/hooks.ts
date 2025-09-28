import { useDeleteMessage, useFeedback } from '@/hooks/chat-hooks';
import { useSetModalState } from '@/hooks/common-hooks';
import { IRemoveMessageById, useSpeechWithSse } from '@/hooks/logic-hooks';
import { IFeedbackRequestBody } from '@/interfaces/request/chat';
// import { getMessagePureId } from '@/utils/chat';
import { pauseAllAudio } from '@/utils/dom-util';
import lzString from 'lz-string';
import { useCallback, useRef, useState } from 'react';

export const useSendFeedback = (messageId: string) => {
  const { visible, hideModal, showModal } = useSetModalState();
  const { feedback, loading } = useFeedback();

  const onFeedbackOk = useCallback(
    async (params: IFeedbackRequestBody) => {
      const ret = await feedback({
        ...params,
        messageId: messageId,
      });

      if (ret === 0) {
        hideModal();
      }
    },
    [feedback, hideModal, messageId],
  );

  return {
    loading,
    onFeedbackOk,
    visible,
    hideModal,
    showModal,
  };
};

export const useRemoveMessage = (
  messageId: string,
  removeMessageById?: IRemoveMessageById['removeMessageById'],
) => {
  const { deleteMessage, loading } = useDeleteMessage();

  const onRemoveMessage = useCallback(async () => {
    if (messageId) {
      const retcode = await deleteMessage(messageId);
      if (retcode === 0) {
        removeMessageById?.(messageId);
      }
    }
  }, [deleteMessage, messageId, removeMessageById]);

  return { onRemoveMessage, loading };
};

export const useSpeech = (content: string, audioBinary?: string) => {
  const ref = useRef<HTMLAudioElement>(null);
  const { read } = useSpeechWithSse();
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [isRequested, setIsRequested] = useState<boolean>(false);

  const pause = useCallback(() => {
    ref.current.currentTime = 0;
    ref.current?.pause();
  }, []);

  const speech = useCallback(async () => {
    // 使用 pako.gzip 压缩文本并将其转换为 Base64
    let content_cleaned = content.replace(/(##\d+\$\$\s*)/g, '').trim();
    // const audioUrl = encodeURI(
    //   process.env.UMI_APP_TTS_URL + `?text=${lzString.compressToBase64(content_cleaned+content_cleaned+content_cleaned)}`,
    // );
    const audioUrl = encodeURI(
      process.env.UMI_APP_TTS_URL +
        `?text=${lzString.compressToEncodedURIComponent(content_cleaned)}`,
    );
    // console.log(audioUrl);
    if (ref.current.src != audioUrl) {
      ref.current.src = audioUrl;
      ref.current.addEventListener('ended', () => setIsPlaying(false));
      ref.current.addEventListener('pause', () => setIsPlaying(false));
      setIsRequested(true);
    }
    ref.current.currentTime = 0;
    ref.current?.play();
  }, [read, content]);

  const handleRead = useCallback(async () => {
    pauseAllAudio();
    if (isPlaying) {
      setIsPlaying(false);
      pause();
    } else {
      setIsPlaying(true);
      speech();
    }
  }, [setIsPlaying, speech, isPlaying, pause]);

  // useEffect(() => {
  //   if (audioBinary) {
  //     const units = hexStringToUint8Array(audioBinary);
  //     if (units) {
  //       try {
  //         player.current?.feed(units);
  //       } catch (error) {
  //         console.warn(error);
  //       }
  //     }
  //   }
  // }, [audioBinary]);

  // useEffect(() => {
  //   initialize();
  // }, [initialize]);

  return { ref, handleRead, isPlaying };
};
