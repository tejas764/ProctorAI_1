from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnrollmentQuestion:
    question_id: str
    text: str


ENROLLMENT_QUESTIONS: tuple[EnrollmentQuestion, ...] = (
    EnrollmentQuestion("Q01", "Please state your full name and candidate ID clearly."),
    EnrollmentQuestion("Q02", "Please read this sentence: Academic integrity matters in every exam."),
    EnrollmentQuestion("Q03", "Please state the name of your course and current semester."),
    EnrollmentQuestion("Q04", "Please read this sentence: I will not use unauthorized help during this session."),
    EnrollmentQuestion("Q05", "Please say the current date and local time."),
    EnrollmentQuestion("Q06", "Please read this sentence: My webcam and microphone will remain active."),
    EnrollmentQuestion("Q07", "Please describe your exam environment in one short sentence."),
    EnrollmentQuestion("Q08", "Please read this sentence: I confirm that I am the registered test taker."),
    EnrollmentQuestion("Q09", "Please count from one to ten in a normal speaking voice."),
    EnrollmentQuestion("Q10", "Please read this sentence: ProctorGuard AI is verifying my voice continuously."),
)

