<p align="center">
  <a href="README.md"><strong>OVERVIEW</strong></a> &nbsp; · &nbsp;
  <a href="DEMO.md"><strong>DEMO & SCREENSHOTS</strong></a> &nbsp; · &nbsp;
  <a href="DEMO.md#demo-setup"><strong>SETUP & DEPLOYMENT</strong></a>
</p>

<div align="center">

# Demo Studio

**ProctorAI_1**

Walk through the experience. Explore the visuals. Prepare your demo.

</div>

---

## Visual preview

> **Preview status:** Fresh application screenshots have not yet been captured. The walkthrough below describes how to prepare the demo.

---

<a id="demo-setup"></a>

## Setup

Follow the repository README for prerequisites and current limitations. Its documented commands are reproduced here for convenience:

```bash
git clone https://github.com/tejas764/ProctorAI_1.git
cd ProctorAI_1
```

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirement.txt
```

```powershell
python web_enrollment_app.py
```

```powershell
python main_pipeline.py
```

```powershell
python chunks.py
```

## Presentation checklist

1. Complete the README setup and resolve its listed missing files or service dependencies.
2. Use an isolated development account and fictional content, such as “Demo User” and “Sample Project”. Do not use real student, customer, or biometric records.
3. Show the main screen, one implemented interaction, and its resulting state.
4. Explain incomplete features and any simulated behavior described in the README.
5. Restore only the test records you created after the presentation.

## Screenshot checklist

Capture the main screen and the completed interaction at desktop and mobile widths where applicable. Store images under `docs/screenshots/`, with a caption containing the tested commit, viewport, and steps. Avoid account identifiers, tokens, and personal data. No fresh verified screenshots are included in this guide.

## Deployment readiness

Prefer a local demonstration with consented sample inputs. Hardware/model/Android requirements make a generic static web deployment unsuitable.

## Verification status

This guide is based on repository documentation. Runtime behavior, screenshots, and hosted availability have not been independently verified in this update.
