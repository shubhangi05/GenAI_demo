from google import genai
import requests
import math

HISTORY = []

client = genai.Client(api_key="YOUR_API_KEY")


def sum_tool(args):
    return args["num1"] + args["num2"]


def prime_tool(args):
    num = args["num"]

    if num < 2:
        return False

    for i in range(2, int(math.sqrt(num)) + 1):
        if num % i == 0:
            return False

    return True


def get_crypto_price(args):
    coin = args["coin"]

    url = f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&ids={coin}"
    response = requests.get(url)

    return response.json()


sum_declaration = {
    "name": "sum",
    "description": "Get the sum of 2 numbers",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "num1": {
                "type": "NUMBER",
                "description": "First number for addition"
            },
            "num2": {
                "type": "NUMBER",
                "description": "Second number for addition"
            }
        },
        "required": ["num1", "num2"]
    }
}


prime_declaration = {
    "name": "prime",
    "description": "Check if a number is prime",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "num": {
                "type": "NUMBER",
                "description": "Number to check"
            }
        },
        "required": ["num"]
    }
}


crypto_declaration = {
    "name": "getCryptoPrice",
    "description": "Get crypto price like bitcoin",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "coin": {
                "type": "STRING",
                "description": "Crypto currency name like bitcoin"
            }
        },
        "required": ["coin"]
    }
}


available_tools = {
    "sum": sum_tool,
    "prime": prime_tool,
    "getCryptoPrice": get_crypto_price
}


def run_agent(user_problem):

    HISTORY.append({
        "role": "user",
        "parts": [{"text": user_problem}]
    })

    while True:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=HISTORY,
            config={
                "system_instruction": """
                You are an AI Agent.
                You have access to three tools:
                - sum of 2 numbers
                - crypto price
                - check prime number
                
                Use these tools whenever required.
                If a question is general you can answer directly.
                """,
                "tools": [{
                    "function_declarations": [
                        sum_declaration,
                        prime_declaration,
                        crypto_declaration
                    ]
                }]
            }
        )

        # parts = response.candidates[0].content.parts

        # call = None

        # for part in parts:
        #     if hasattr(part, "function_call") and part.function_call:
        #         call = part.function_call
        #         break

        # if call:

        #     name = call.name
        #     args = dict(call.args)

        #     print("Tool Called:", name, args)

        #     func = available_tools[name]
        #     result = func(args)

        #     print("Tool Called:", name, args)

        #     func = available_tools[name]
        #     result = func(args)

        #     HISTORY.append({
        #         "role": "model",
        #         "parts": [{
        #             "function_call": {
        #                 "name": name,
        #                 "args": args
        #             }
        #         }]
        #     })

        #     HISTORY.append({
        #         "role": "user",
        #         "parts": [{
        #             "function_response": {
        #                 "name": name,
        #                 "response": {"result": result}
        #             }
        #         }]
        #     })

        # else:

        #     text = response.text
        #     HISTORY.append({
        #         "role": "model",
        #         "parts": [{"text": text}]
        #     })

        #     print(text)
        #     break

        parts = response.candidates[0].content.parts

        tool_called = False

        for part in parts:

            if hasattr(part, "function_call") and part.function_call:

                tool_called = True

                call = part.function_call
                name = call.name
                args = dict(call.args)

                print("Tool Called:", name, args)

                func = available_tools[name]
                result = func(args)

                HISTORY.append({
                    "role": "model",
                    "parts": [{
                        "function_call": {
                            "name": name,
                            "args": args
                        }
                    }]
                })

                HISTORY.append({
                    "role": "user",
                    "parts": [{
                        "function_response": {
                            "name": name,
                            "response": {"result": result}
                        }
                    }]
                })


        if not tool_called:

            text = response.text

            HISTORY.append({
                "role": "model",
                "parts": [{"text": text}]
            })

            print(text)
            break


def main():
    while True:
        user_problem = input("Ask me anything --> ")
        run_agent(user_problem)


if __name__ == "__main__":
    main()