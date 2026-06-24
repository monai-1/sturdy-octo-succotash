import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# ===================== 页面基础配置 =====================
st.set_page_config(
    page_title="学生成绩管理与可视化分析系统",
    page_icon="📊",
    layout="wide"
)

# ===================== 常量定义 =====================
DATA_DIR = "data/user_data"
DATA_PATH = os.path.join(DATA_DIR, "grades.csv")
REQUIRED_COLUMNS = ["学号", "姓名", "班级", "科目", "学分", "成绩"]

# 自动创建数据目录
os.makedirs(DATA_DIR, exist_ok=True)

# ===================== 核心工具函数 =====================
# --- 数据处理模块 ---
def validate_grade_data(df: pd.DataFrame) -> tuple[bool, str]:
    if df.empty:
        return False, "数据为空"
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        return False, f"缺少必填列：{', '.join(missing_cols)}"
    if df["学号"].isna().any() or df["学号"].astype(str).str.strip().eq("").any():
        return False, "存在空学号，请检查数据"
    if df["学分"].isna().any() or (df["学分"] <= 0).any():
        return False, "学分必须为大于0的数字"
    if df["成绩"].isna().any():
        return False, "存在空成绩，请检查数据"
    if (df["成绩"] < 0).any() or (df["成绩"] > 100).any():
        return False, "成绩必须在0-100范围内"
    return True, "校验通过"

def clean_grade_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["学号", "科目"], keep="last").reset_index(drop=True)
    for col in ["学号", "姓名", "班级", "科目"]:
        df[col] = df[col].astype(str).str.strip()
    df["学分"] = pd.to_numeric(df["学分"], errors="coerce")
    df["成绩"] = pd.to_numeric(df["成绩"], errors="coerce")
    return df

def load_grade_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    df = pd.read_csv(DATA_PATH)
    return clean_grade_data(df)

def save_grade_data(df: pd.DataFrame) -> None:
    df.to_csv(DATA_PATH, index=False, encoding="utf-8-sig")

def get_template_df() -> pd.DataFrame:
    template_data = [
        ["2024001", "张三", "计算机2401班", "高等数学", 4, 85],
        ["2024001", "张三", "计算机2401班", "Python程序设计", 3, 92],
        ["2024002", "李四", "计算机2401班", "高等数学", 4, 58],
    ]
    return pd.DataFrame(template_data, columns=REQUIRED_COLUMNS)

def get_sample_data() -> pd.DataFrame:
    sample_data = [
        ["2024001", "张三", "计算机2401班", "高等数学", 4, 85],
        ["2024001", "张三", "计算机2401班", "Python程序设计", 3, 92],
        ["2024001", "张三", "计算机2401班", "大学英语", 2, 78],
        ["2024001", "张三", "计算机2401班", "数据结构", 4, 88],
        ["2024002", "李四", "计算机2401班", "高等数学", 4, 58],
        ["2024002", "李四", "计算机2401班", "Python程序设计", 3, 72],
        ["2024002", "李四", "计算机2401班", "大学英语", 2, 65],
        ["2024002", "李四", "计算机2401班", "数据结构", 4, 61],
        ["2024003", "王五", "计算机2401班", "高等数学", 4, 95],
        ["2024003", "王五", "计算机2401班", "Python程序设计", 3, 98],
        ["2024003", "王五", "计算机2401班", "大学英语", 2, 82],
        ["2024003", "王五", "计算机2401班", "数据结构", 4, 91],
        ["2024004", "赵六", "计算机2401班", "高等数学", 4, 73],
        ["2024004", "赵六", "计算机2401班", "Python程序设计", 3, 68],
        ["2024004", "赵六", "计算机2401班", "大学英语", 2, 75],
        ["2024004", "赵六", "计算机2401班", "数据结构", 4, 70],
    ]
    return pd.DataFrame(sample_data, columns=REQUIRED_COLUMNS)

# --- 成绩计算模块 ---
def score_to_gpa(score: float, scale: str = "4分制") -> float:
    if pd.isna(score) or score < 60:
        return 0.0
    if scale == "4分制":
        return min(4.0, 4.0 - (90 - score) * 0.1) if score <= 90 else 4.0
    elif scale == "5分制":
        return min(5.0, 5.0 - (90 - score) * 0.1) if score <= 90 else 5.0
    return 0.0

def calc_weighted_gpa(df: pd.DataFrame, scale: str = "4分制") -> float:
    if df.empty:
        return 0.0
    df = df.copy()
    df["单科学分绩点"] = df.apply(lambda x: score_to_gpa(x["成绩"], scale) * x["学分"], axis=1)
    total_credit = df["学分"].sum()
    return round(df["单科学分绩点"].sum() / total_credit, 2) if total_credit > 0 else 0.0

def calc_class_rank(grade_df: pd.DataFrame, student_id: str, scale: str = "4分制") -> tuple[int, int]:
    if grade_df.empty:
        return 0, 0
    student_gpas = grade_df.groupby(["学号", "姓名", "班级"]).apply(
        lambda x: calc_weighted_gpa(x, scale)
    ).reset_index(name="绩点")
    student_gpas = student_gpas.sort_values("绩点", ascending=False).reset_index(drop=True)
    match = student_gpas[student_gpas["学号"] == student_id]
    if match.empty:
        return 0, len(student_gpas)
    return int(match.index[0] + 1), len(student_gpas)

def calc_class_indicators(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"平均分": 0, "及格率": "0%", "优秀率": "0%", "挂科人数": 0, "总人数": 0}
    student_avg = df.groupby("学号")["成绩"].mean()
    total = len(student_avg)
    pass_count = len(student_avg[student_avg >= 60])
    excellent_count = len(student_avg[student_avg >= 90])
    fail_count = total - pass_count
    return {
        "平均分": round(student_avg.mean(), 1),
        "及格率": f"{round(pass_count / total * 100, 1)}%",
        "优秀率": f"{round(excellent_count / total * 100, 1)}%",
        "挂科人数": fail_count,
        "总人数": total
    }

# --- 可视化模块 ---
def plot_score_distribution(df: pd.DataFrame, subject: str = "全部科目") -> go.Figure:
    if subject != "全部科目":
        df = df[df["科目"] == subject]
    fig = px.histogram(
        df, x="成绩", nbins=10, color_discrete_sequence=["#1f77b4"],
        title=f"{subject} 成绩分布直方图",
        labels={"成绩": "分数", "count": "人数"}
    )
    fig.update_layout(bargap=0.1, title_x=0.5)
    fig.add_vline(x=60, line_dash="dash", line_color="red", annotation_text="及格线")
    return fig

def plot_subject_average(df: pd.DataFrame) -> go.Figure:
    subject_avg = df.groupby("科目")["成绩"].mean().reset_index()
    subject_avg = subject_avg.sort_values("成绩", ascending=False)
    subject_avg["成绩"] = subject_avg["成绩"].round(1)
    fig = px.bar(
        subject_avg, x="科目", y="成绩", text="成绩",
        color="成绩", color_continuous_scale="Blues",
        title="各科平均分对比"
    )
    fig.update_layout(title_x=0.5, showlegend=False)
    fig.update_traces(textposition="outside")
    return fig

def plot_student_radar(student_df: pd.DataFrame) -> go.Figure:
    fig = px.line_polar(
        student_df, r="成绩", theta="科目", line_close=True,
        range_r=[0, 100], title="个人各科成绩雷达图",
        color_discrete_sequence=["#ff7f0e"]
    )
    fig.update_layout(title_x=0.5)
    fig.update_traces(fill='toself')
    return fig

def plot_score_segment_pie(df: pd.DataFrame) -> go.Figure:
    bins = [0, 60, 70, 80, 90, 101]
    labels = ["不及格(0-59)", "及格(60-69)", "中等(70-79)", "良好(80-89)", "优秀(90-100)"]
    df = df.copy()
    df["分数段"] = pd.cut(df["成绩"], bins=bins, labels=labels, right=False)
    segment = df["分数段"].value_counts().reset_index()
    segment.columns = ["分数段", "人数"]
    fig = px.pie(
        segment, names="分数段", values="人数",
        title="班级分数段占比", hole=0.3,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(title_x=0.5)
    return fig

# ===================== 侧边栏导航 =====================
st.sidebar.title("📊 功能导航")
page = st.sidebar.radio(
    "选择功能模块",
    ["系统首页", "成绩录入", "个人绩点查询", "班级统计分析", "报表导出"]
)
st.sidebar.markdown("---")
st.sidebar.caption("大学生期末作业项目")

# ===================== 页面逻辑 =====================
# 1. 系统首页
if page == "系统首页":
    st.title("📊 学生成绩管理与可视化分析系统")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("系统功能说明")
        st.write("✅ 支持单条录入与批量导入成绩数据")
        st.write("✅ 支持4分制/5分制学分加权绩点计算")
        st.write("✅ 自动计算班级排名与超越比例")
        st.write("✅ 多维度成绩统计可视化图表")
        st.write("✅ 一键导出Excel格式成绩报表")
        
        st.markdown("---")
        st.subheader("快速开始")
        if st.button("一键生成示例数据（快速体验）", type="primary"):
            sample_df = get_sample_data()
            save_grade_data(sample_df)
            st.success("示例数据已生成，可前往各功能模块体验")
            st.rerun()
    
    with col2:
        st.info("💡 技术栈")
        st.write("- Python 3.8+")
        st.write("- Streamlit 前端框架")
        st.write("- Pandas 数据处理")
        st.write("- Plotly 可视化图表")

# 2. 成绩录入
elif page == "成绩录入":
    st.title("📝 成绩录入与导入")
    existing_df = load_grade_data()

    tab1, tab2 = st.tabs(["单条录入", "批量导入"])

    with tab1:
        st.subheader("单条成绩录入")
        with st.form("single_entry_form"):
            col1, col2, col3 = st.columns(3)
            student_id = col1.text_input("学号")
            name = col2.text_input("姓名")
            class_name = col3.text_input("班级")
            
            col4, col5 = st.columns(2)
            subject = col4.text_input("科目")
            credit = col5.number_input("学分", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
            score = st.slider("成绩", min_value=0, max_value=100, value=75)
            
            submitted = st.form_submit_button("录入成绩", use_container_width=True)
            
            if submitted:
                if not all([student_id, name, class_name, subject]):
                    st.error("请填写完整信息")
                else:
                    new_row = pd.DataFrame([[student_id, name, class_name, subject, credit, score]],
                                         columns=REQUIRED_COLUMNS)
                    updated_df = pd.concat([existing_df, new_row], ignore_index=True)
                    updated_df = clean_grade_data(updated_df)
                    save_grade_data(updated_df)
                    st.success("成绩录入成功！")
                    st.rerun()

    with tab2:
        st.subheader("批量文件导入")
        template_df = get_template_df()
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            template_df.to_excel(writer, index=False, sheet_name="成绩模板")
        st.download_button(
            label="📥 下载导入模板（Excel）",
            data=buffer.getvalue(),
            file_name="成绩导入模板.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        uploaded_file = st.file_uploader("上传成绩文件（支持CSV/Excel）", type=["csv", "xlsx"])
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith(".csv"):
                    upload_df = pd.read_csv(uploaded_file)
                else:
                    upload_df = pd.read_excel(uploaded_file)
                
                is_valid, msg = validate_grade_data(upload_df)
                if not is_valid:
                    st.error(f"数据校验失败：{msg}")
                else:
                    upload_df = clean_grade_data(upload_df)
                    st.success(f"数据校验通过，共 {len(upload_df)} 条记录")
                    st.dataframe(upload_df, use_container_width=True, height=300)
                    
                    if st.button("确认导入并保存", type="primary", use_container_width=True):
                        merged_df = pd.concat([existing_df, upload_df], ignore_index=True)
                        merged_df = clean_grade_data(merged_df)
                        save_grade_data(merged_df)
                        st.success("批量导入成功！")
                        st.rerun()
            except Exception as e:
                st.error(f"文件解析失败：{str(e)}")

    st.markdown("---")
    st.subheader("当前已保存数据预览")
    if existing_df.empty:
        st.info("暂无已保存数据，请先录入或导入")
    else:
        st.dataframe(existing_df, use_container_width=True)
        if st.button("清空所有数据", type="secondary"):
            save_grade_data(pd.DataFrame(columns=REQUIRED_COLUMNS))
            st.rerun()

# 3. 个人绩点查询
elif page == "个人绩点查询":
    st.title("🔍 个人绩点与排名查询")
    df = load_grade_data()

    if df.empty:
        st.warning("暂无成绩数据，请先前往成绩录入页面导入数据")
    else:
        student_id = st.text_input("请输入学号")
        scale = st.radio("绩点计算标准", ["4分制", "5分制"], horizontal=True)

        if st.button("查询", use_container_width=True) and student_id:
            student_df = df[df["学号"] == student_id]
            
            if student_df.empty:
                st.error("未找到该学号的成绩信息")
            else:
                student_name = student_df["姓名"].iloc[0]
                student_class = student_df["班级"].iloc[0]
                gpa = calc_weighted_gpa(student_df, scale)
                rank, total = calc_class_rank(df, student_id, scale)
                
                st.subheader(f"{student_name}（{student_class}）")
                col1, col2, col3 = st.columns(3)
                col1.metric("加权绩点", gpa)
                col2.metric("班级排名", f"{rank}/{total}")
                col3.metric("超越比例", f"{round((1 - rank/total)*100, 1)}%" if total > 0 else "0%")
                
                st.markdown("---")
                st.subheader("各科成绩明细")
                st.dataframe(student_df[["科目", "学分", "成绩"]], use_container_width=True, hide_index=True)
                
                st.plotly_chart(plot_student_radar(student_df), use_container_width=True)

# 4. 班级统计分析
elif page == "班级统计分析":
    st.title("📊 班级成绩统计分析")
    df = load_grade_data()

    if df.empty:
        st.warning("暂无成绩数据，请先前往成绩录入页面导入数据")
    else:
        class_list = ["全部班级"] + sorted(df["班级"].unique().tolist())
        selected_class = st.sidebar.selectbox("选择班级", class_list)
        subject_list = ["全部科目"] + sorted(df["科目"].unique().tolist())
        selected_subject = st.sidebar.selectbox("选择科目", subject_list)

        filter_df = df.copy()
        if selected_class != "全部班级":
            filter_df = filter_df[filter_df["班级"] == selected_class]
        if selected_subject != "全部科目":
            filter_df = filter_df[filter_df["科目"] == selected_subject]

        indicators = calc_class_indicators(filter_df)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("平均分", indicators["平均分"])
        col2.metric("及格率", indicators["及格率"])
        col3.metric("优秀率", indicators["优秀率"])
        col4.metric("挂科人数", indicators["挂科人数"])

        st.markdown("---")

        col_left, col_right = st.columns(2)
        with col_left:
            st.plotly_chart(plot_score_distribution(filter_df, selected_subject), use_container_width=True)
            st.plotly_chart(plot_score_segment_pie(filter_df), use_container_width=True)
        with col_right:
            st.plotly_chart(plot_subject_average(filter_df), use_container_width=True)
            
            st.subheader("班级绩点TOP10")
            rank_df = filter_df.groupby(["学号", "姓名"]).apply(
                lambda x: calc_weighted_gpa(x, "4分制")
            ).reset_index(name="绩点")
            rank_df = rank_df.sort_values("绩点", ascending=False).head(10).reset_index(drop=True)
            rank_df.index = rank_df.index + 1
            st.dataframe(rank_df, use_container_width=True)

# 5. 报表导出
elif page == "报表导出":
    st.title("📤 成绩报表导出")
    df = load_grade_data()

    if df.empty:
        st.warning("暂无成绩数据，请先录入数据")
    else:
        export_type = st.radio("导出内容", ["原始成绩明细表", "学生绩点汇总表"], horizontal=True)
        scale = st.selectbox("绩点计算标准", ["4分制", "5分制"])

        if export_type == "原始成绩明细表":
            export_df = df
        else:
            summary = df.groupby(["学号", "姓名", "班级"]).apply(
                lambda x: pd.Series({
                    "总学分": x["学分"].sum(),
                    "加权绩点": calc_weighted_gpa(x, scale),
                    "平均分": round(x["成绩"].mean(), 1)
                })
            ).reset_index()
            summary = summary.sort_values("加权绩点", ascending=False).reset_index(drop=True)
            summary.insert(0, "排名", range(1, len(summary)+1))
            export_df = summary

        st.subheader("导出预览")
        st.dataframe(export_df, use_container_width=True, hide_index=True)

        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            export_df.to_excel(writer, index=False, sheet_name="成绩报表")

        st.download_button(
            label="📥 导出为Excel文件",
            data=buffer.getvalue(),
            file_name=f"成绩报表_{export_type}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            type="primary"
        )
