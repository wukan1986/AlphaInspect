from typing import Optional, List

import lightgbm as lgb
import pandas as pd
import seaborn as sns

html_template = r"""
<script src="https://unpkg.com/d3@5.16.0/dist/d3.min.js"></script>
<script src="https://unpkg.com/@hpcc-js/wasm@0.3.11/dist/index.min.js"></script>
<script src="https://unpkg.com/d3-graphviz@3.0.5/build/d3-graphviz.min.js"></script>
<div id="graph" style="text-align: center;"></div>
<script>
var dotSrc = `
{{dotSrc}}
`;

d3.select("#graph")
  .graphviz()
    .renderDot(dotSrc);
</script>
"""


def plot_importance_box(models, plot_top_n: int = 20, importance_type: str = 'gain', ax=None):
    """多树模型特征重要性"""
    if len(models) == 1:
        lgb.plot_importance(models[0], importance_type=importance_type, max_num_features=plot_top_n, ax=ax)
        return

    # when plot_top_n < 0, the last plot_top features will be shown
    assert plot_top_n != 0

    importance_df = []
    for i, model in enumerate(models):
        importance = model.feature_importance(importance_type)
        feature_name = model.feature_name()
        importance_df.append(dict(zip(feature_name, importance)))
    importance_df = pd.DataFrame(importance_df)

    # sort features by mean
    sorted_indices = importance_df.mean(axis=0).sort_values(ascending=False).index
    sorted_importance_df = importance_df.loc[:, sorted_indices]
    # plot top N features
    if plot_top_n > 0:
        plot_cols = sorted_importance_df.columns[:plot_top_n]
    else:
        plot_cols = sorted_importance_df.columns[plot_top_n:]

    ax.grid()
    ax.set_xscale('log')
    ax.set_ylabel('Features')
    ax.set_xlabel('Feature importance')
    sns.boxplot(data=sorted_importance_df[plot_cols],
                orient='h',
                ax=ax)


def plot_metric_errorbar(models, metric: str, ax=None):
    """多树模型评估函数图

    Notes
    -----
    evals_result = {}  # to record eval results for plotting
    callbacks = [
        record_evaluation(evals_result),
    ]

    # 非常重要，否则训练结果不保存
    model.evals_result_ = deepcopy(evals_result)

    """
    if len(models) == 1:
        lgb.plot_metric(models[0].evals_result_, ax=ax, metric=metric)
        return

    # 提取指标
    results = {}
    for i, model in enumerate(models):
        results[i] = {k: v[metric] for k, v in model.evals_result_.items()}

    # 位置调整
    out = {}
    for k, v in results.items():
        for k1, v1 in v.items():
            if k1 not in out:
                out[k1] = []
            out[k1].append(v1)

    # 画图
    for k, v in out.items():
        d = pd.DataFrame(v)
        ax.errorbar(x=d.columns, y=d.mean(), yerr=d.std(), label=k, lw=1, alpha=0.6)
    ax.grid()
    ax.set_title('Metric during training')
    ax.set_xlabel('Iterations')
    ax.set_ylabel(metric)
    ax.legend()


def tree_to_html(
        booster,
        tree_index: int = 0,
        show_info: Optional[List[str]] = ['split_gain', 'leaf_count', 'data_percentage'],
        precision: Optional[int] = 3,
        orientation: str = 'horizontal',
        example_case=None,
        **kwargs):
    """
    将树转换为html

    lgb.plot_tree显示的图观察实在不方便，而dtreeviz这个库性能又有点慢，所以实现了一个简版

    with open('tree.html', 'w', encoding='utf-8') as f:
        f.write(tree_to_html(models[-1], tree_index=0, example_case=X_test.iloc[[10], :]))
        os.system(f'"tree.html"')

    from IPython.display import HTML, display
    display(HMTL(tree_to_html(models[-1], tree_index=0, example_case=X_test.iloc[[10], :])))

    在jupyter notebook中，需要trust才能显示网页。在VSCode jupyter中目前还无法正常显示。

    """
    graph = lgb.create_tree_digraph(booster=booster, tree_index=tree_index,
                                    show_info=show_info, precision=precision,
                                    orientation=orientation, example_case=example_case, **kwargs)
    return html_template.replace('{{dotSrc}}', f'{graph}')
