export const scrollToBottom = (element: HTMLElement) => {
  element.scrollTo(0, element.scrollHeight);
};

export const pauseAllAudio = () => {
  try {
    const audioElements = document.querySelectorAll('audio');
    audioElements.forEach((audio) => audio.pause());
  } catch (error) {
    // 如果出现错误，可能是用户拒绝了权限请求或者设备不可用
    console.error('Error pauseAllAudio:', error);
  }
};
