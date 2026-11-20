"use strict";
const $ = (selector) => document.querySelector(selector);
const names = {
  small: "偏小",
  fit: "合身",
  large: "偏大",
  sizing: "尺码",
  material: "面料",
  length: "长度",
};
const modelNames = {
  majority: "多数类基线",
  metadata_logistic: "元数据 · Logistic",
  history_logistic: "历史特征 · Logistic",
  history_mlp: "历史特征 · MLP",
};
const state = { items: [], metrics: null };
function element(tag, text, className) {
  const e = document.createElement(tag);
  if (text !== undefined) e.textContent = text;
  if (className) e.className = className;
  return e;
}
function status(text, error = false) {
  $("#status").textContent = text;
  $("#status").className = error ? "error" : "";
}
async function api(path, body) {
  const response = await fetch(
    path,
    body === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "请求失败");
  return data;
}
function percent(value) {
  return (100 * value).toFixed(1) + "%";
}
function currentItem() {
  return state.items.find((i) => i.item_id === $("#item-select").value);
}
function showItem() {
  const item = currentItem();
  if (!item) return;
  $("#item-title").textContent = item.category + " / #" + item.item_id;
  $("#item-count").textContent =
    item.count.toLocaleString() +
    " 条历史反馈 · " +
    Object.keys(item.sizes).length +
    " 种已观察到的尺码编码";
  const distribution = $("#fit-distribution");
  distribution.replaceChildren();
  for (const label of ["small", "fit", "large"]) {
    const ratio = item.labels[label] / item.count;
    const part = element("span", ratio > 0.07 ? percent(ratio) : "");
    part.style.width = percent(ratio);
    part.style.background = "var(--" + label + ")";
    part.title = names[label] + " " + percent(ratio);
    distribution.append(part);
  }
  const aspects = $("#aspect-bars");
  aspects.replaceChildren();
  for (const aspect of ["sizing", "fit", "material", "length"]) {
    const ratio = item.negative_aspects[aspect] / item.count;
    const row = element("div", undefined, "aspect-row");
    const track = element("div", undefined, "bar-track");
    const fill = element("div", undefined, "bar-fill");
    fill.style.width = percent(ratio);
    track.append(fill);
    row.append(
      element("span", aspect === "fit" ? "版型" : names[aspect]),
      track,
      element("strong", percent(ratio)),
    );
    aspects.append(row);
  }
  $("#history-cutoff").textContent =
    "历史截止日期：" + state.metrics.splits.train.end;
  $("#fit-result").replaceChildren();
}
async function predict(event) {
  if (event) event.preventDefault();
  const button = $("#predict-btn");
  button.disabled = true;
  button.textContent = "计算中…";
  try {
    const result = await api("/api/recommend", {
      item_id: $("#item-select").value,
      height_cm: Number($("#height").value),
      weight_kg: Number($("#weight").value),
      body_type: $("#body-type").value,
    });
    const container = $("#fit-result");
    container.replaceChildren();
    const heading = element(
      "div",
      "当前最高合身概率：编码 ",
      "recommend-heading",
    );
    heading.append(element("strong", result.recommended_size_code));
    container.append(heading);
    for (const c of result.candidates) {
      const row = element(
        "div",
        undefined,
        "candidate" +
          (c.size_code === result.recommended_size_code ? " best" : ""),
      );
      const track = element("div", undefined, "bar-track");
      const fill = element("div", undefined, "bar-fill");
      fill.style.width = percent(c.probabilities.fit);
      track.append(fill);
      row.append(
        element("span", "编码 " + c.size_code),
        track,
        element("span", percent(c.probabilities.fit)),
      );
      row.title =
        "偏小 " +
        percent(c.probabilities.small) +
        " · 合身 " +
        percent(c.probabilities.fit) +
        " · 偏大 " +
        percent(c.probabilities.large);
      container.append(row);
    }
    container.append(
      element(
        "p",
        "仅比较该商品至少出现 3 次的尺码；概率未经校准，不保证实际穿着结果。",
        "fine-print",
      ),
    );
    status("");
  } catch (error) {
    status(error.message, true);
  } finally {
    button.disabled = false;
    button.textContent = "比较尺码候选 →";
  }
}
async function extract(event) {
  event.preventDefault();
  const button = $("#extract-btn");
  button.disabled = true;
  button.textContent = "分析中…";
  try {
    const data = await api("/api/extract", {
      review: $("#review-text").value,
      engine: $("#engine").value,
    });
    const container = $("#review-results");
    container.replaceChildren();
    if (!data.findings.length)
      container.append(
        element(
          "div",
          "没有找到可支持的方面。试试描述面料、尺码或长度的英文评论。",
          "empty",
        ),
      );
    for (const finding of data.findings) {
      const card = element("div", undefined, "evidence");
      const head = element("div", undefined, "evidence-head");
      head.append(
        element("b", finding.aspect === "fit" ? "版型" : names[finding.aspect]),
        element(
          "span",
          { positive: "正向", negative: "负向", neutral: "中性" }[
            finding.sentiment
          ],
          "sentiment " + finding.sentiment,
        ),
      );
      card.append(head, element("blockquote", "“" + finding.evidence + "”"));
      container.append(card);
    }
    container.append(
      element(
        "small",
        data.engine === "rules"
          ? "规则基线 · 原文引用已保留"
          : "LLM 抽取 · 原文引用已校验",
        "fine-print",
      ),
    );
    status("");
  } catch (error) {
    status(error.message, true);
  } finally {
    button.disabled = false;
    button.textContent = "提取洞察 ↗";
  }
}
async function start() {
  try {
    const [overview, catalog] = await Promise.all([
      api("/api/overview"),
      api("/api/catalog"),
    ]);
    state.items = catalog.items;
    state.metrics = overview.metrics;
    const m = state.metrics;
    $("#metric-rows").textContent = m.cleaning.retained_rows.toLocaleString();
    $("#metric-f1").textContent =
      m.models[m.selected_model].test.macro_f1.toFixed(3);
    $("#metric-model").textContent = modelNames[m.selected_model];
    $("#metric-items").textContent = overview.catalog_size;
    for (const item of state.items) {
      const option = element(
        "option",
        item.category + " · #" + item.item_id + " · " + item.count + " 条反馈",
      );
      option.value = item.item_id;
      $("#item-select").append(option);
    }
    for (const name of [
      "majority",
      "metadata_logistic",
      "history_logistic",
      "history_mlp",
    ]) {
      const result = m.models[name];
      const row = element(
        "tr",
        undefined,
        name === m.selected_model ? "selected" : "",
      );
      row.append(
        element("td", modelNames[name] || name),
        element("td", result.validation.macro_f1.toFixed(3)),
        element("td", result.test.macro_f1.toFixed(3)),
        element("td", percent(result.test.accuracy)),
        element("td", name === m.selected_model ? "✓ 当前模型" : "对照"),
      );
      $("#model-table").append(row);
    }
    for (const name of ["train", "validation", "test"]) {
      const split = m.splits[name];
      const div = element(
        "div",
        { train: "训练集", validation: "验证集", test: "测试集" }[name] +
          " · " +
          split.rows.toLocaleString() +
          " 条",
      );
      div.append(element("small", split.start + " → " + split.end));
      $("#split-summary").append(div);
    }
    if (overview.llm_available) {
      $("#engine option[value=llm]").disabled = false;
      $("#engine option[value=llm]").textContent =
        "LLM · 将评论发送到已配置服务";
    }
    showItem();
    status("");
    await predict();
  } catch (error) {
    status("无法加载工作台：" + error.message, true);
  }
}
$("#fit-form").addEventListener("submit", predict);
$("#review-form").addEventListener("submit", extract);
$("#item-select").addEventListener("change", showItem);
start();
