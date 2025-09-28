import { Spin } from 'antd';
import { useEffect, useState } from 'react';
import styles from './index.less';
import './pptx.css';

const Pptx = ({ filePath }: { filePath: string }) => {
  const [succeed, setSucceed] = useState<boolean>(false);
  const pptxJsUrls = [
    './js/jquery-1.11.3.min.js',
    './js/jszip.min.js',
    './js/filereader.js',
    './js/d3.min.js',
    './js/nv.d3.min.js',
    './js/pptxjs.js',
    './js/divs2slides.js',
    './js/jquery.fullscreen-min.js',
    './js/postFullscreen.js',
  ];

  useEffect(() => {
    try {
      pptxJsUrls.forEach((url: any, index: number) => {
        const script = document.createElement('script');
        console.log(require(`${url}`));
        const textNode = document.createTextNode(require(`${url}`).default);
        script.appendChild(textNode);
        script.type = 'text/javascript';
        script.async = false;
        document.body.appendChild(script);
      });

      const script = document.createElement('script');
      const scriptContent = `
        $("#slide-resolte-contaniner").pptxToHtml({
          pptxFileUrl: "${filePath}",
          slideMode: false,
          keyBoardShortCut: false
        });
      `;
      const textNode = document.createTextNode(scriptContent);
      script.appendChild(textNode);
      script.async = false;
      document.body.appendChild(script);

      setSucceed(true);
    } catch (e) {
      console.error(e);
    }

    return () => {
      // document.body.removeChild(script);
    };
  }, []);

  return (
    <>
      <div style={{display: succeed ? 'block':'none'}} id="slide-resolte-contaniner"></div>

      <section className={styles.docxViewerWrapper}>
        <div style={{display: succeed ? 'none':'none'}} id="pptx" className={styles.box}>
          <Spin />
        </div>
      </section>
    </>
  );
};

export default Pptx;
