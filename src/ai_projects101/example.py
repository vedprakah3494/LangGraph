def understand_request(request):
    return request.lower()

def choose_action(request):
    if "delete" in request:
        return "Ask human for approval"
    return "Continue with the safe workflow"

request=understand_request("Delete an Old Account")
action=choose_action(request)
print(action)

