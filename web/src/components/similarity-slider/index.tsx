import { useTranslate } from '@/hooks/common-hooks';
import { Form, Slider } from 'antd';

type FieldType = {
  similarity_threshold?: number;
  // vector_similarity_weight?: number;
  search_depth?: number;
};

interface IProps {
  isTooltipShown?: boolean;
  isSearchDepthShown?: boolean;
  vectorSimilarityWeightName?: string;
}

const SimilaritySlider = ({
  isTooltipShown = false,
  isSearchDepthShown = true,
  vectorSimilarityWeightName = 'vector_similarity_weight',
}: IProps) => {
  const { t } = useTranslate('knowledgeDetails');

  return (
    <>
      <Form.Item<FieldType>
        label={t('similarityThreshold')}
        name={'similarity_threshold'}
        tooltip={isTooltipShown && t('similarityThresholdTip')}
        initialValue={0.2}
      >
        <Slider max={1} step={0.01} />
      </Form.Item>
      <Form.Item
        // style={{ display: 'none' }}
        label={t('vectorSimilarityWeight')}
        name={vectorSimilarityWeightName}
        initialValue={0.4}
        tooltip={isTooltipShown && t('vectorSimilarityWeightTip')}
      >
        <Slider max={1} step={0.01} />
      </Form.Item>
      <Form.Item<FieldType>
        label={t('searchDepth')}
        name={'search_depth'}
        tooltip={isTooltipShown && t('searchDepthTip')}
        initialValue={0.2}
        style={{ display: isSearchDepthShown ? 'block' : 'none' }}
      >
        <Slider min={0.1} max={1} step={0.1} />
      </Form.Item>
    </>
  );
};

export default SimilaritySlider;
