import { Images } from '@/constants/common';
import { api_host } from '@/utils/api';
import { Flex, Image } from 'antd';
import { useParams, useSearchParams } from 'umi';
import Docx from './docx';
import Excel from './excel';
import Pdf from './pdf';
import Pptx from './pptx';
import Csv from './csv';
import Html from './html';
import styles from './index.less';
// TODO: The interface returns an incorrect content-type for the SVG.

const DocumentViewer = () => {
  const { id: documentId } = useParams();
  const [currentQueryParameters] = useSearchParams();
  const ext = currentQueryParameters.get('ext');
  const prefix = currentQueryParameters.get('prefix');
  const api = `${api_host}/${prefix || 'file'}/get/${documentId}`;

  return (
    <section className={styles.viewerWrapper}>
      {Images.includes(ext!) && (
        <Flex className={styles.image} align="center" justify="center">
          <Image src={api} preview={false}></Image>
        </Flex>
      )}

      {ext === 'pdf' && <Pdf url={api}></Pdf>}

      {(ext === 'xlsx' || ext === 'xls') && <Excel filePath={api}></Excel>}

      {ext === 'docx' && <Docx filePath={api}></Docx>}

      {(ext === 'pptx' || ext === 'ppt') && <Pptx filePath={api} />}

      {ext === 'csv' && <Csv filePath={api}></Csv>}

      {(ext === 'htm' || ext === 'html') && <Html filePath={api}></Html>}
    </section>
  );
};

export default DocumentViewer;
