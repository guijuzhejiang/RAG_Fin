
def print_search_results_and_scores(sres, sim, tsim, vsim, title_suffix=""):
    """统一的打印函数：显示搜索结果前20个单词和对应的相似度分数"""
    print(f"=== 搜索结果和相似度分数 {title_suffix}===")
    for i, (doc_id, s, t, v) in enumerate(zip(sres.ids, sim, tsim, vsim)):
        # 从field字段获取文档内容
        content = ""
        if sres.field and doc_id in sres.field:
            content = str(sres.field[doc_id].get('content_with_weight', ''))

        # 提取前20个单词
        words = content.split()[:20]
        content_preview = " ".join(words)
        if len(content.split()) > 20:
            content_preview += "..."

        print(f"文档 { i +1} (ID: {doc_id})")
        print(f"  内容: {content_preview}")
        print(f"  sim: {s:.4f}, tsim: {t:.4f}, vsim: {v:.4f}")
        print()