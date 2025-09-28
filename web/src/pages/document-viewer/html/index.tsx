import { Spin } from 'antd';
import React, { useEffect, useState } from 'react';
import styles from './index.less';

const Html = ({ filePath }: { filePath: string }) => {
  const [succeed, setSucceed] = useState<boolean>(false);
  const [blobUrl, setBlobUrl] = useState('');

  useEffect(() => {
    try {
      fetch(filePath)
        .then((response) => response.text())
        .then((data) => {
          const blob = new Blob([data], { type: 'text/html' });
          const url = URL.createObjectURL(blob);
          setBlobUrl(url);
        })
        .catch((error) => console.error('Error fetching HTML:', error));

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
      {succeed ? (
        <iframe
          src={blobUrl}
          title="HTML Content"
          style={{ width: '100%', height: '100%', border: 'none' }}
        />
      ) : (
        <section className={styles.docxViewerWrapper}>
          <div id="html" className={styles.box}>
            <Spin />
          </div>
        </section>
      )}
    </>
  );
};

export default Html;
