import { EmptyConversationId } from '@/constants/chat';
import { Message } from '@/interfaces/database/chat';
import { IMessage } from '@/pages/chat/interface';
import { v4 as uuid } from 'uuid';

export const isConversationIdExist = (conversationId: string) => {
  return conversationId !== EmptyConversationId && conversationId !== '';
};

export const buildMessageUuid = (message: Partial<Message | IMessage>) => {
  if ('id' in message && message.id) {
    // return message.role === MessageType.User
    //   ? `${MessageType.User}_${message.id}`
    //   : `${MessageType.Assistant}_${message.id}`;
    return message.id;
  }
  return uuid();
};

// export const getMessagePureId = (id?: string) => {
//   const strings = id?.split('_') ?? [];
//   if (strings.length > 0) {
//     return strings.at(-1);
//   }
//   return id;
// };

export const buildMessageListWithUuid = (messages?: Message[]) => {
  return (
    messages?.map((x: Message | IMessage) => ({
      ...x,
      id: buildMessageUuid(x),
    })) ?? []
  );
};

export const getConversationId = () => {
  return uuid().replace(/-/g, '');
};

// When rendering each message, add a prefix to the id to ensure uniqueness.
export const buildMessageUuidWithRole = (
  message: Partial<Message | IMessage>,
) => {
  return `${message.role}_${message.id}`;
};

// Preprocess LaTeX equations to be rendered by KaTeX
// ref: https://github.com/remarkjs/react-markdown/issues/785

export const preprocessLaTeX = (content: string) => {
  const blockProcessedContent = content.replace(
    /\\\[([\s\S]*?)\\\]/g,
    (_, equation) => `$$${equation}$$`,
  );
  const inlineProcessedContent = blockProcessedContent.replace(
    /\\\(([\s\S]*?)\\\)/g,
    (_, equation) => `$${equation}$`,
  );
  return inlineProcessedContent;
};
