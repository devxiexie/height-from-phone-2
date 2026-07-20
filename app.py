import gradio as gr

import phone_specs
from pipeline import analyze


def _report_markdown(res):
    if not res.ok:
        return f"### ❌ Could not analyze\n\n{res.message}"

    inch = res.height_cm / 2.54
    ft_whole = int(inch // 12)
    in_rem = round(inch - ft_whole * 12)
    if in_rem == 12:
        ft_whole, in_rem = ft_whole + 1, 0
    unc_in = res.uncertainty_cm / 2.54

    basis_label = {
        "full body": "full body (feet visible)",
        "head": "partial body, estimated from head size",
    }.get(res.basis, f"partial body, extrapolated from {res.basis}")

    lines = [
        f"## Estimated height: **{ft_whole}′{in_rem:.0f}″  ± {unc_in:.0f}″**",
        f"### {res.height_cm:.0f} cm  ± {res.uncertainty_cm:.0f} cm",
        "",
        "| | |",
        "|---|---|",
        f"| Phone reference | {res.phone_source} (width = {res.phone_width_px:.0f}px) |",
        f"| Height basis | {basis_label} |",
    ]
    if res.phone_depth_applied:
        lines.append(
            f"| Depth correction | applied: phone was ~{(1-res.depth_factor)*100:.0f}% closer to the mirror |"
        )
        
    else:
        lines.append("| Depth correction | not applied (see notes) |")

    lines.append("")
    
    if res.notes:
        lines.append("**Notes:**")
        for n in res.notes:
            lines.append(f"- {n}")

    lines.append("")
    return "\n".join(lines)


def run(image, phone_choice, custom_h, custom_w):
    if image is None:
        return None, None, "### Please upload a photo first."
    h_mm, w_mm, source = phone_specs.resolve_phone_size(
        phone_choice, custom_h, custom_w
    )
    res = analyze(image, h_mm, w_mm, source)
    return res.annotated, res.depth_map, _report_markdown(res)


def build_ui():
    with gr.Blocks(title="Height from Phone") as demo:
        gr.Markdown(
            "Upload a mirror selfie that shows a **person** and their **phone**. "
            "The photo of the person should either contain their whole body or just their head "
        )
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="pil", label="Mirror selfie", height=380)
                phone_choice = gr.Dropdown(
                    choices=phone_specs.dropdown_choices(),
                    value=phone_specs.GENERIC_LABEL,
                    label="Phone model (scale reference)",
                )
                with gr.Row():
                    custom_h = gr.Number(label="Custom height (mm)", value=None)
                    custom_w = gr.Number(label="Custom width (mm)", value=None)
                gr.Markdown(
                    "_Pick your exact phone for the best accuracy. Choose "
                    "**Custom** and type the body height in mm if it's not "
                    "listed, or leave it on **Generic**._"
                )
                btn = gr.Button("Analyze", variant="primary")
            with gr.Column(scale=1):
                out_annotated = gr.Image(label="Detections + height", height=380)
                out_depth = gr.Image(label="Depth map (bright = closer)", height=300)
        report = gr.Markdown()

        btn.click(
            run,
            inputs=[image, phone_choice, custom_h, custom_w],
            outputs=[out_annotated, out_depth, report],
        )
    return demo


if __name__ == "__main__":
    build_ui().launch()
