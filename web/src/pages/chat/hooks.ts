import { ChatSearchParams, MessageType } from '@/constants/chat';
import { fileIconMap } from '@/constants/common';
import {
  useFetchManualConversation,
  useFetchManualDialog,
  useFetchNextConversation,
  useFetchNextConversationList,
  useFetchNextDialog,
  useGetChatSearchParams,
  useRemoveNextConversation,
  useRemoveNextDialog,
  useSetNextDialog,
  useUpdateNextConversation,
} from '@/hooks/chat-hooks';
import {
  useSetModalState,
  useShowDeleteConfirm,
  useTranslate,
} from '@/hooks/common-hooks';
import {
  useRegenerateMessage,
  useSelectDerivedMessages,
  useSendMessageWithSse,
} from '@/hooks/logic-hooks';
import { IConversation, IDialog, Message } from '@/interfaces/database/chat';
import { getFileExtension } from '@/utils';
import api from '@/utils/api';
import { getConversationId } from '@/utils/chat';
import { pauseAllAudio } from '@/utils/dom-util';
import { useMicVAD, utils } from '@ray8716397/vad-react';
import { useMutationState } from '@tanstack/react-query';
import { get } from 'lodash';
import trim from 'lodash/trim';
import {
  ChangeEventHandler,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { useSearchParams } from 'umi';
import { v4 as uuid } from 'uuid';
import {
  IClientConversation,
  IMessage,
  VariableTableDataType,
} from './interface';

export const useSetChatRouteParams = () => {
  const [currentQueryParameters, setSearchParams] = useSearchParams();
  const newQueryParameters: URLSearchParams = useMemo(
    () => new URLSearchParams(currentQueryParameters.toString()),
    [currentQueryParameters],
  );

  const setConversationIsNew = useCallback(
    (value: string) => {
      newQueryParameters.set(ChatSearchParams.isNew, value);
      setSearchParams(newQueryParameters);
    },
    [newQueryParameters, setSearchParams],
  );

  const getConversationIsNew = useCallback(() => {
    return newQueryParameters.get(ChatSearchParams.isNew);
  }, [newQueryParameters]);

  return { setConversationIsNew, getConversationIsNew };
};

export const useSetNewConversationRouteParams = () => {
  const [currentQueryParameters, setSearchParams] = useSearchParams();
  const newQueryParameters: URLSearchParams = useMemo(
    () => new URLSearchParams(currentQueryParameters.toString()),
    [currentQueryParameters],
  );

  const setNewConversationRouteParams = useCallback(
    (conversationId: string, isNew: string) => {
      newQueryParameters.set(ChatSearchParams.ConversationId, conversationId);
      newQueryParameters.set(ChatSearchParams.isNew, isNew);
      setSearchParams(newQueryParameters);
    },
    [newQueryParameters, setSearchParams],
  );

  return { setNewConversationRouteParams };
};

export const useSelectCurrentDialog = () => {
  const data = useMutationState({
    filters: { mutationKey: ['fetchDialog'] },
    select: (mutation) => {
      return get(mutation, 'state.data.data', {});
    },
  });

  return (data.at(-1) ?? {}) as IDialog;
};

export const useSelectPromptConfigParameters = (): VariableTableDataType[] => {
  const { data: currentDialog } = useFetchNextDialog();

  const finalParameters: VariableTableDataType[] = useMemo(() => {
    const parameters = currentDialog?.prompt_config?.parameters ?? [];
    if (!currentDialog.id) {
      // The newly created chat has a default parameter
      return [{ key: uuid(), variable: 'knowledge', optional: true }];
    }
    return parameters.map((x) => ({
      key: uuid(),
      variable: x.key,
      optional: x.optional,
    }));
  }, [currentDialog]);

  return finalParameters;
};

export const useDeleteDialog = () => {
  const showDeleteConfirm = useShowDeleteConfirm();

  const { removeDialog } = useRemoveNextDialog();

  const onRemoveDialog = (dialogIds: Array<string>) => {
    showDeleteConfirm({ onOk: () => removeDialog(dialogIds) });
  };

  return { onRemoveDialog };
};

export const useHandleItemHover = () => {
  const [activated, setActivated] = useState<string>('');

  const handleItemEnter = (id: string) => {
    setActivated(id);
  };

  const handleItemLeave = () => {
    setActivated('');
  };

  return {
    activated,
    handleItemEnter,
    handleItemLeave,
  };
};

export const useEditDialog = () => {
  const [dialog, setDialog] = useState<IDialog>({} as IDialog);
  const { fetchDialog } = useFetchManualDialog();
  const { setDialog: submitDialog, loading } = useSetNextDialog();

  const {
    visible: dialogEditVisible,
    hideModal: hideDialogEditModal,
    showModal: showDialogEditModal,
  } = useSetModalState();

  const hideModal = useCallback(() => {
    setDialog({} as IDialog);
    hideDialogEditModal();
  }, [hideDialogEditModal]);

  const onDialogEditOk = useCallback(
    async (dialog: IDialog) => {
      const ret = await submitDialog(dialog);

      if (ret === 0) {
        hideModal();
      }
    },
    [submitDialog, hideModal],
  );

  const handleShowDialogEditModal = useCallback(
    async (dialogId?: string) => {
      if (dialogId) {
        const ret = await fetchDialog(dialogId);
        if (ret.retcode === 0) {
          setDialog(ret.data);
        }
      }
      showDialogEditModal();
    },
    [showDialogEditModal, fetchDialog],
  );

  const clearDialog = useCallback(() => {
    setDialog({} as IDialog);
  }, []);

  return {
    dialogSettingLoading: loading,
    initialDialog: dialog,
    onDialogEditOk,
    dialogEditVisible,
    hideDialogEditModal: hideModal,
    showDialogEditModal: handleShowDialogEditModal,
    clearDialog,
  };
};

//#region conversation

export const useSelectDerivedConversationList = () => {
  const { t } = useTranslate('chat');

  const [list, setList] = useState<Array<IConversation>>([]);
  const { data: currentDialog } = useFetchNextDialog();
  const { data: conversationList, loading } = useFetchNextConversationList();
  const { dialogId } = useGetChatSearchParams();
  const prologue = currentDialog?.prompt_config?.prologue ?? '';
  const { setNewConversationRouteParams } = useSetNewConversationRouteParams();

  const addTemporaryConversation = useCallback(() => {
    const conversationId = getConversationId();
    setList((pre) => {
      if (dialogId) {
        setNewConversationRouteParams(conversationId, 'true');
        const nextList = [
          {
            id: conversationId,
            name: t('newConversation'),
            dialog_id: dialogId,
            is_new: true,
            message: [
              {
                content: prologue,
                role: MessageType.Assistant,
              },
            ],
          } as any,
          ...conversationList,
        ];
        return nextList;
      }

      return pre;
    });
  }, [conversationList, dialogId, prologue, t, setNewConversationRouteParams]);

  // When you first enter the page, select the top conversation card

  useEffect(() => {
    setList([...conversationList]);
  }, [conversationList]);

  useEffect(() => {
    // 无对话时自动添加临时对话
    if (!loading && conversationList.length === 0) {
      addTemporaryConversation();
    }
  }, [conversationList, loading]);

  return { list, addTemporaryConversation, loading };
};

export const useSetConversation = () => {
  const { dialogId } = useGetChatSearchParams();
  const { updateConversation } = useUpdateNextConversation();

  const setConversation = useCallback(
    async (
      message: string,
      isNew: boolean = false,
      conversationId?: string,
    ) => {
      const data = await updateConversation({
        dialog_id: dialogId,
        name: message,
        is_new: isNew,
        conversation_id: conversationId,
        message: [
          {
            role: MessageType.Assistant,
            content: message,
          },
        ],
      });

      return data;
    },
    [updateConversation, dialogId],
  );

  return { setConversation };
};

export const useSelectNextMessages = () => {
  const {
    ref,
    setDerivedMessages,
    derivedMessages,
    addNewestAnswer,
    addNewestQuestion,
    removeLatestMessage,
    removeMessageById,
    removeMessagesAfterCurrentMessage,
  } = useSelectDerivedMessages();
  const { data: conversation, loading } = useFetchNextConversation();
  const { data: dialog } = useFetchNextDialog();
  const { conversationId, dialogId, isNew } = useGetChatSearchParams();

  const addPrologue = useCallback(() => {
    if (dialogId !== '' && isNew === 'true') {
      const prologue = dialog.prompt_config?.prologue;

      const nextMessage = {
        role: MessageType.Assistant,
        content: prologue,
        id: uuid(),
      } as IMessage;

      setDerivedMessages([nextMessage]);
    }
  }, [isNew, dialog, dialogId, setDerivedMessages]);

  useEffect(() => {
    addPrologue();
  }, [addPrologue]);

  useEffect(() => {
    if (
      conversationId &&
      isNew !== 'true' &&
      conversation.message?.length > 0
    ) {
      setDerivedMessages(conversation.message);
    }

    if (!conversationId) {
      setDerivedMessages([]);
    }
  }, [conversation.message, conversationId, setDerivedMessages, isNew]);

  return {
    ref,
    derivedMessages,
    loading,
    addNewestAnswer,
    addNewestQuestion,
    removeLatestMessage,
    removeMessageById,
    removeMessagesAfterCurrentMessage,
  };
};

export const useHandleMessageInputChange = () => {
  const [value, setValue] = useState('');

  const handleInputChange: ChangeEventHandler<HTMLInputElement> = (e) => {
    const value = e.target.value;
    const nextValue = value.replaceAll('\\n', '\n').replaceAll('\\t', '\t');
    setValue(nextValue);
  };

  return {
    handleInputChange,
    value,
    setValue,
  };
};

export const useSendNextMessage = (controller: AbortController) => {
  const { setConversation } = useSetConversation();
  const { conversationId, isNew } = useGetChatSearchParams();
  const { handleInputChange, value, setValue } = useHandleMessageInputChange();

  const { send, answer, done } = useSendMessageWithSse(
    api.completeConversation,
  );
  const {
    ref,
    derivedMessages,
    loading,
    addNewestAnswer,
    addNewestQuestion,
    removeLatestMessage,
    removeMessageById,
    removeMessagesAfterCurrentMessage,
  } = useSelectNextMessages();
  const { setConversationIsNew, getConversationIsNew } =
    useSetChatRouteParams();

  const sendMessage = useCallback(
    async ({
      message,
      currentConversationId,
      messages,
    }: {
      message: Message;
      currentConversationId?: string;
      messages?: Message[];
    }) => {
      const res = await send(
        {
          conversation_id: currentConversationId ?? conversationId,
          messages: [...(messages ?? derivedMessages ?? []), message],
        },
        controller,
      );

      if (res && (res?.response.status !== 200 || res?.data?.retcode !== 0)) {
        // cancel loading
        setValue(message.content);
        console.info('removeLatestMessage111');
        removeLatestMessage();
      }
    },
    [
      derivedMessages,
      conversationId,
      removeLatestMessage,
      setValue,
      send,
      controller,
    ],
  );

  const handleSendMessage = useCallback(
    async (message: Message) => {
      const isNew = getConversationIsNew();
      if (isNew !== 'true') {
        sendMessage({ message });
      } else {
        const data = await setConversation(
          message.content,
          true,
          conversationId,
        );
        if (data.retcode === 0) {
          setConversationIsNew('');
          const id = data.data.id;

          const params = new URLSearchParams(window.location.search);

          // 修改或添加 URL 参数
          params.set('conversationId', id);

          // 生成新的 URL
          const newUrl = `${window.location.pathname}?${params.toString()}`;

          // 使用 replaceState 替换当前的 URL，但不刷新页面
          window.history.replaceState(null, '', newUrl);

          // currentConversationIdRef.current = id;
          sendMessage({
            message,
            currentConversationId: id,
            messages: data.data.message,
          });
        }
      }
    },
    [
      setConversation,
      sendMessage,
      setConversationIsNew,
      getConversationIsNew,
      conversationId,
    ],
  );

  const { regenerateMessage } = useRegenerateMessage({
    removeMessagesAfterCurrentMessage,
    sendMessage,
    messages: derivedMessages,
  });

  useEffect(() => {
    //  #1289
    const params = new URLSearchParams(window.location.search);

    if (
      answer.answer &&
      (answer?.conversationId === conversationId ||
        answer?.conversationId === params.get('conversationId'))
    ) {
      addNewestAnswer(answer);
    }
  }, [answer, addNewestAnswer, conversationId, isNew]);

  const handlePressEnter = useCallback(
    (documentIds: string[]) => {
      if (trim(value) === '') return;
      const id = uuid();

      addNewestQuestion({
        content: value,
        doc_ids: documentIds,
        id,
        role: MessageType.User,
      });
      if (done) {
        setValue('');
        handleSendMessage({
          id,
          content: value.trim(),
          role: MessageType.User,
          doc_ids: documentIds,
        });
      }
    },
    [addNewestQuestion, handleSendMessage, done, setValue, value],
  );

  return {
    handlePressEnter,
    handleInputChange,
    value,
    setValue,
    regenerateMessage,
    sendLoading: !done,
    loading,
    ref,
    derivedMessages,
    removeMessageById,
  };
};

export const useGetFileIcon = () => {
  const getFileIcon = (filename: string) => {
    const ext: string = getFileExtension(filename);
    const iconPath = fileIconMap[ext as keyof typeof fileIconMap];
    return `@/assets/svg/file-icon/${iconPath}`;
  };

  return getFileIcon;
};

export const useDeleteConversation = () => {
  const showDeleteConfirm = useShowDeleteConfirm();
  const { removeConversation } = useRemoveNextConversation();

  const deleteConversation = (conversationIds: Array<string>) => async () => {
    const ret = await removeConversation(conversationIds);

    return ret;
  };

  const onRemoveConversation = (conversationIds: Array<string>) => {
    showDeleteConfirm({ onOk: deleteConversation(conversationIds) });
  };

  return { onRemoveConversation };
};

export const useRenameConversation = () => {
  const [conversation, setConversation] = useState<IClientConversation>(
    {} as IClientConversation,
  );
  const { fetchConversation } = useFetchManualConversation();
  const {
    visible: conversationRenameVisible,
    hideModal: hideConversationRenameModal,
    showModal: showConversationRenameModal,
  } = useSetModalState();
  const { updateConversation, loading } = useUpdateNextConversation();

  const onConversationRenameOk = useCallback(
    async (name: string) => {
      const ret = await updateConversation({
        ...conversation,
        conversation_id: conversation.id,
        name,
        is_new: false,
      });

      if (ret.retcode === 0) {
        hideConversationRenameModal();
      }
    },
    [updateConversation, conversation, hideConversationRenameModal],
  );

  const handleShowConversationRenameModal = useCallback(
    async (conversationId: string) => {
      const ret = await fetchConversation(conversationId);
      if (ret.retcode === 0) {
        setConversation(ret.data);
      }
      showConversationRenameModal();
    },
    [showConversationRenameModal, fetchConversation],
  );

  return {
    conversationRenameLoading: loading,
    initialConversationName: conversation.name,
    onConversationRenameOk,
    conversationRenameVisible,
    hideConversationRenameModal,
    showConversationRenameModal: handleShowConversationRenameModal,
  };
};

export const useGetSendButtonDisabled = () => {
  const { dialogId, conversationId } = useGetChatSearchParams();

  return dialogId === '' || conversationId === '';
};

export const useSendButtonDisabled = (value: string) => {
  return trim(value) === '';
};

export const useCreateConversationBeforeUploadDocument = () => {
  const { setConversation } = useSetConversation();
  const { dialogId } = useGetChatSearchParams();

  const createConversationBeforeUploadDocument = useCallback(
    async (message: string) => {
      const data = await setConversation(message, true);

      return data;
    },
    [setConversation],
  );

  return {
    createConversationBeforeUploadDocument,
    dialogId,
  };
};
//#endregion

// STT
export const useSTT = (
  messageInput,
  setMessageInput,
  messageInputRef,
  handlePressEnter,
) => {
  // mic 是否打开
  const [micOn, setMicOn] = useState(false);
  const [micAvailable, setMicAvailable] = useState(false);
  // 是否正在处理STT
  const [STTIng, setSTTIng] = useState(false);
  // 持续讲话模式
  const [voiceContinuationEnable, setVoiceContinuationEnable] = useState(false);
  // wav float32数组缓存
  const [audioBuffer, setAudioBuffer] = useState([]);
  // 1min有没有讲话
  const [userSpeakLately, setUserSpeakLately] = useState<Date>(new Date());
  const timerRef = useRef(null);
  const [timerInterval, setTimerInterval] = useState<any>(null);
  const vadTimeoutMS = 120 * 1000;
  // 3s 自动语音3s没有说话
  const voiceContinueTimerRef = useRef(null);
  const [voiceContinueTimerInterval, setVoiceContinueTimerInterval] =
    useState<any>(null);
  const voiceContinueTimeoutMS = 1 * 1000;

  const vad = useMicVAD({
    workletURL: '/vad/vad.worklet.bundle.min.js',
    modelURL: '/vad/silero_vad.onnx',
    startOnLoad: true,
    startOnInit: false,
    positiveSpeechThreshold: 0.8,
    negativeSpeechThreshold: 0.8 - 0.15,
    minSpeechFrames: 3,
    preSpeechPadFrames: 1,
    redemptionFrames: parseInt(String(8)),
    // onS
    onVADMisfire: () => {
      console.log('onVADMisfire');
      setSTTIng(false);
    },
    onSpeechStart: () => {
      try {
        console.log('onSpeechStart');
        setUserSpeakLately(new Date());
        pauseAllAudio();

        setSTTIng(true);
      } catch (e) {
        console.error('onSpeechStart error:' + e);
      }
    },
    onSpeechEnd: (float32Audio) => {
      try {
        console.log('onSpeechEnd');
        setAudioBuffer((prevItems) => [
          ...prevItems,
          utils.encodeWAV(float32Audio),
        ]);

        clearInterval(voiceContinueTimerRef.current);
      } catch (e) {
        console.error('onSpeechEnd error:' + e);
      }
    },
  });

  // 无说话关闭监测
  useEffect(() => {
    if (timerInterval) {
      setTimerInterval(false);
      const curDate = new Date();
      if (userSpeakLately) {
        const diffInMilliseconds = Math.abs(
          userSpeakLately.getTime() - curDate.getTime(),
        );
        if (diffInMilliseconds > vadTimeoutMS) {
          vad.stop();
          setMicOn(false);
          setVoiceContinuationEnable(false);
          clearInterval(timerRef.current);
          clearInterval(voiceContinueTimerRef.current);
        }
      }
    }
  }, [userSpeakLately, timerInterval]);

  // 自动语音无说话关闭监测
  useEffect(() => {
    // console.log("trigger")
    if (voiceContinueTimerInterval) {
      setVoiceContinueTimerInterval(false);
      const curDate = new Date();
      if (userSpeakLately) {
        const diffInMilliseconds = Math.abs(
          userSpeakLately.getTime() - curDate.getTime(),
        );

        if (messageInput) {
          handlePressEnter();
        } else {
          clearInterval(voiceContinueTimerRef.current);
        }
      }
    }
  }, [userSpeakLately, voiceContinueTimerInterval]);

  // mic输入的wav监测
  useEffect(() => {
    if (audioBuffer.length > 0) {
      const formData = new FormData();
      audioBuffer.map((wavBuf, i) => {
        const wavBlob = new Blob([wavBuf], { type: 'audio/wav' });
        formData.append('audio_file', wavBlob, 'audio.wav');
      });
      // clear buffer
      setAudioBuffer([]);

      const startTime = performance.now();
      fetch(process.env.UMI_APP_STT_URL, {
        method: 'POST',
        headers: {
          accept: 'application/json',
        },
        body: formData,
      })
        .then((response) => {
          if (response.ok) {
            return response.text();
          } else {
            console.error('Failed to upload');
          }
        })
        .then((data) => {
          if (typeof data === 'string') {
            // alert("22"+voiceContinuationEnable);
            if (!voiceContinuationEnable) {
              setMessageInput(messageInput + data);
              const input = messageInputRef?.current.input;
              // console.log(input);
              // console.log(input.value.length)
              if (input) {
                // 聚焦输入框并将光标移到文本末尾
                input.focus();
                input.setSelectionRange(input.value.length, input.value.length);
              }
            } else {
              setMessageInput(messageInput + data);
              handlePressEnter();
              // 3s检查input有没有
              voiceContinueTimerRef.current = setInterval(() => {
                setVoiceContinueTimerInterval(true);
              }, voiceContinueTimeoutMS);
            }
          } else {
            console.error('setVoiceText failed');
          }

          console.log('stt elapsed ' + (performance.now() - startTime) + 'ms');
        })
        .catch((error) => {
          setAudioBuffer([]);
          console.error('上传错误:', error);
        })
        .finally(() => {
          setSTTIng(false);
        });
    } else {
      // console.log('State changed in silenceDurationMS seconds!!!!!!!!!!!!!!!!!');
    }
  }, [audioBuffer]);

  useEffect(() => {}, []);

  return {
    micOn,
    setMicOn,
    micAvailable,
    setMicAvailable,
    STTIng,
    setSTTIng,
    voiceContinuationEnable,
    setVoiceContinuationEnable,
    userSpeakLately,
    setUserSpeakLately,
    timerRef,
    setTimerInterval,
    vadTimeoutMS,
    voiceContinueTimerRef,
    voiceContinueTimerInterval,
    setVoiceContinueTimerInterval,
    voiceContinueTimeoutMS,
    vad,
  };
};
