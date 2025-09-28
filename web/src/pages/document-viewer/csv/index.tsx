import { Spin } from 'antd';
import Papa from "papaparse";
import "react-table/react-table.css";
import ReactTable from "react-table";
import React, { useEffect, useState } from 'react';
import styles from './index.less';

const Csv = ({ filePath }: { filePath: string }) => {
  const [succeed, setSucceed] = useState<boolean>(false);
  const [data, setData] = useState([]);
  const [columns, setColumns] = useState([]);

  const makeColumns = rawColumns => {
    return rawColumns.map(column => {
      return { Header: column, accessor: column };
    });
  };

  const handleDataChange = file => {
    setData(file.data);
    setColumns(makeColumns(file.meta.fields));
  };

  useEffect(() => {
    try {
      // const xhr = new XMLHttpRequest();
      // xhr.open('GET', filePath, false); // false 表示同步请求
      // xhr.send();
      //
      // const blob = xhr.response;

      // 创建 File 对象
      // const fileName = filePath.split('/').pop(); // 从 URL 获取文件名
      // const fileObject = new File([blob], fileName+'.csv', { type: blob.type });
      Papa.parse(filePath, {
        header: true,
        download: true,
        dynamicTyping: true,
        complete: handleDataChange
      });

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
        <ReactTable
          data={data}
          columns={columns}
          defaultPageSize={10}
          className="-striped -highlight"
        />
      ) : (
        <section className={styles.docxViewerWrapper}>
          <div id="csv" className={styles.box}>
            <Spin />
          </div>
        </section>
      )}
    </>
  );
};

export default Csv;
