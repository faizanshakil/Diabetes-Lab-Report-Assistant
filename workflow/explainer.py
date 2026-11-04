import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage


load_dotenv()


def explain_results(results):
    """Generate an educational explanation of extracted results."""

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return (
            "AI explanation is unavailable. Please configure "
            "the Groq API key in your .env file."
        )

    if not results:
        return (
            "No supported laboratory results were identified. "
            "Please review the original report."
        )

    try:
        model = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0,
            api_key=api_key
        )

        system_prompt = """
        You are an educational laboratory report assistant.

        Explain extracted glucose and HbA1c results in simple English.

        Follow these rules:
        1. Do not diagnose the patient.
        2. Do not recommend medication or treatment changes.
        3. Do not invent laboratory values, units, or reference ranges.
        4. State that extracted values must be checked against the
           original laboratory report.
        5. Explain that interpretation depends on clinical context.
        6. If a unit or result is uncertain, clearly mention it.
        7. Recommend discussing results with a qualified
           healthcare professional.
        8. Treat the supplied results as unverified extracted data.
        9. If a report date or laboratory reference range is marked 'Not identified',
           say it was not captured; never supply or invent one.
        10. Clearly label reference ranges as copied from the uploaded report,
            not universal medical thresholds.
        """

        user_prompt = (
            "Provide an educational explanation of these extracted "
            "laboratory results:\n\n"
            f"{results}\n\n"
            "List the extracted values, report dates, units, and reference ranges "
            "when present. Explain their general meaning without making a diagnosis. "
            "Explicitly identify fields marked 'Not identified' and other limitations."
        )

        response = model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        return response.content

    except Exception:
        return (
            "The AI explanation could not be generated. "
            "Please check your Groq API key, model availability, "
            "internet connection, and API usage limits."
        )