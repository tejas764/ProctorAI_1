from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnrollmentQuestion:
    question_id: str
    text: str


ENROLLMENT_QUESTIONS: tuple[EnrollmentQuestion, ...] = (
    EnrollmentQuestion(
        "Q01",
        "I authorize the storage of my voice data for calculating reference mean values, and I confirm that I will not use unethical methods to outsmart the system.",
    ),
    EnrollmentQuestion("Q02", "This project was developed by the Centre for AI at CHRIST University."),
    EnrollmentQuestion(
        "Q03",
        "This web application uses gaze detection, lip-sync detection, and object detection techniques.",
    ),
)

