import streamlit as st
import ast

st.set_page_config(page_title="AI Robot Programming Tutor", layout="centered")

st.title("🤖 AI-Based Robot Programming Tutor")
st.write("Upload your robot Python program and get expert feedback on safety, control, and efficiency.")

uploaded_file = st.file_uploader("Upload Robot Python Code", type=["py"])

SAFE_SPEED_LIMIT = 1.5
SAFE_KP_LIMIT = 5.0
MAX_JUMP_ANGLE = 90

# ---------- Helper function to safely get numeric values ----------
def get_number(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    return None

# ---------- ANALYSIS FUNCTION ----------
def analyze_code(code_text):
    feedback = []
    score = 100

    try:
        tree = ast.parse(code_text)
    except Exception as e:
        return ["❌ Code parsing failed: " + str(e)], 0

    class CodeVisitor(ast.NodeVisitor):

        def visit_While(self, node):
            if isinstance(node.test, ast.Constant) and node.test.value in [True, 1]:
                feedback.append("⚠ Infinite loop detected. Add stopping condition.")
            self.generic_visit(node)

        def visit_Assign(self, node):
            for target in node.targets:
                if isinstance(target, ast.Name):

                    value = get_number(node.value)

                    # Speed check
                    if "speed" in target.id.lower() and value is not None:
                        if value > SAFE_SPEED_LIMIT:
                            feedback.append(f"⚠ Speed too high ({value}). Reduce below {SAFE_SPEED_LIMIT}.")

                    # PID Kp check
                    if target.id.lower() == "kp" and value is not None:
                        if value > SAFE_KP_LIMIT:
                            feedback.append(f"⚠ Kp too high ({value}). May cause oscillations.")

            self.generic_visit(node)

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name):

                # Sudden joint movement
                if node.func.id.lower() == "move_joint":
                    if len(node.args) >= 2:
                        angle = get_number(node.args[1])
                        if angle is not None and abs(angle) > MAX_JUMP_ANGLE:
                            feedback.append(f"⚠ Large joint angle jump ({angle}°). Motion may be jerky.")

                # Gripper safety
                if node.func.id.lower() == "close_gripper":
                    feedback.append("⚠ Add small delay before closing gripper for stability.")

            self.generic_visit(node)

    visitor = CodeVisitor()
    visitor.visit(tree)

    score -= len(feedback) * 10
    score = max(score, 0)

    return feedback, score


# ---------- UI OUTPUT ----------
if uploaded_file is not None:
    code = uploaded_file.read().decode("utf-8")

    with st.expander("📄 View Uploaded Code"):
        st.code(code, language="python")

    feedback, score = analyze_code(code)

    st.subheader("🧠 Expert Feedback Report")

    if feedback:
        for f in feedback:
            st.write(f)
    else:
        st.success("✅ No major issues detected. Good job!")

    st.subheader("📊 Program Quality Score")
    st.progress(score / 100)
    st.write(f"### Score: {score}/100")

else:
    st.info("Upload a Python robot control program to begin analysis.")
