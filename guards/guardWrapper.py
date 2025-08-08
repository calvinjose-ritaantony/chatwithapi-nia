# import functools
# from fastapi import Request
# from fastapi.responses import JSONResponse
# from guardrails import Guard
# import logging
# from guardrails.hub import (
#     DetectJailbreak,
#     ToxicLanguage,
#     SecretsPresent,
#     GuardrailsPII
# )
# from dependencies import NiaAzureOpenAIClient
# import re
# import os
# import json
# os.environ["GUARDRAILS_DISABLE_REMOTE_INFERENCING"] = "true"
# os.environ["OTEL_TRACES_EXPORTER"] = "none"
# os.environ["OTEL_SDK_DISABLED"] = "true"

# logger = logging.getLogger(__name__)

# # Initialize input and output guards lazily
# def get_input_guards():
#     return [
#         Guard().use(DetectJailbreak, threshold=0.8, on_fail="exception"),
#         Guard().use(ToxicLanguage, threshold=0.5, validation_method="sentence", on_fail="exception"),
#         Guard().use(SecretsPresent, on_fail="exception")
#     ]

# def get_output_guards():
#     return [
#         Guard().use(ToxicLanguage, threshold=0.5, validation_method="sentence", on_fail="exception"),
#         Guard().use(SecretsPresent, on_fail="exception"),
#         Guard().use(GuardrailsPII, entities=["EMAIL_ADDRESS", "PHONE_NUMBER"], on_fail="exception")
#     ]



# def resolve_guard_name(guard) -> str:
#     try:
#         if hasattr(guard, "validators") and guard.validators:
#             reference = guard.validators[0]
#             logger.info(f"Resolving guard name, reference object: {reference}")

#             # Try to get validator class name
#             if hasattr(reference, "validator") and reference.validator is not None:
#                 logger.info(f"Reference has 'validator' attribute: {reference.validator}")
#                 return reference.validator.__class__.__name__

#             # Try to extract id from string representation
#             ref_str = str(reference)
#             logger.debug(f"Reference string: {ref_str}")
#             match = re.search(r"id='([^']+)'", ref_str)
#             if match:
#                 guard_id = match.group(1)  # e.g., "guardrails/guardrails_pii"
#                 guard_name = guard_id.split("/")[-1]  # get "guardrails_pii"
#                 logger.info(f"Extracted guard name from id: {guard_name}")
#                 return guard_name
#     except Exception as e:
#         logger.error(f"Error resolving guard name: {e}")

#     # Fallback
#     fallback_name = getattr(guard, "id", None) or getattr(guard, "name", "UnknownGuard")
#     logger.info(f"Falling back to guard id or name: {fallback_name}")
#     return fallback_name



# def guardrails_validation(prompt_arg="user_message", response_key="response"):
#     def decorator(func):
#         @functools.wraps(func)
#         async def wrapper(*args, **kwargs):
#             validation_results = []
#             try:
#                 request: Request = kwargs.get("request") or next((a for a in args if isinstance(a, Request)), None)
#                 prompt = kwargs.get(prompt_arg)
#                 if not prompt and request:
#                     form = await request.form()
#                     prompt = form.get(prompt_arg)
#                 logger.info(f"Input prompt extracted: {prompt!r}")

#                 for guard in get_input_guards():
#                     logger.info(f"guard name is {guard.name}, guard description is {guard.description}, guard id is {guard.id}") 
#                     guard_name = resolve_guard_name(guard)
#                     try:
#                         guard.parse(prompt)
#                         validation_results.append({"type": guard_name, "direction": "input", "passed": True})
#                     except Exception as e:
#                         logger.error(f"Guardrails blocked the input: {e}")
#                         validation_results.append({"type": guard_name, "direction": "input", "passed": False, "message": str(e)})
#                         return JSONResponse({"error": f"Guardrails blocked the input:\n{str(e)}", "guardrail_status": validation_results}, status_code=422)
#             except Exception as e:
#                 logger.exception("Unexpected error during input guard processing.")
#                 return JSONResponse({"error": str(e)}, status_code=500)

#             result = await func(*args, **kwargs)

#             response = None
#             if isinstance(result, dict):
#                 response = result.get(response_key)
#             elif hasattr(result, "body"):
#                 try:
#                     body = result.body
#                     if isinstance(body, bytes):
#                         body = body.decode("utf-8")
#                     body_json = json.loads(body)
#                     response = body_json.get(response_key)
#                 except Exception as e:
#                     logger.error(f"Failed to parse JSONResponse body: {e}")
#             logger.info(f"Extracted response: {response}")

#             try:
#                 for guard in get_output_guards():
#                     logger.info(f"guard name is {guard.name}, guard description is {guard.description}") 
#                     guard_name = resolve_guard_name(guard)
#                     try:
#                         guard.validate(response)
#                         validation_results.append({"type": guard_name, "direction": "output", "passed": True})
#                     except Exception as e:
#                         logger.error(f"Guardrails blocked the output: {e}")
#                         validation_results.append({"type": guard_name, "direction": "output", "passed": False, "message": str(e)})
#                         return JSONResponse({"error": f"Guardrails blocked the output:\n{str(e)}", "guardrail_status": validation_results}, status_code=422)
#             except Exception as e:
#                 logger.exception("Unexpected error during output guard processing.")
#                 return JSONResponse({"error": str(e)}, status_code=500)
#             logger.info(f"Guardrails validation results: {validation_results}")
#             return JSONResponse(
#                 {
#                     "response": response,
#                     "total_tokens": result.get("total_tokens", 0) if isinstance(result, dict) else 0,
#                     "follow_up_questions": result.get("follow_up_questions", []) if isinstance(result, dict) else [],
#                     "guardrail_status": validation_results
#                 },
#                 status_code=200
#             )
#         return wrapper
#     return decorator
