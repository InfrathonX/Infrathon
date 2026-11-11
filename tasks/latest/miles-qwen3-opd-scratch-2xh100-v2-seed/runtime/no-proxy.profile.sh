# Mounted into Work at /etc/profile.d. A ChatGPT-login run gives only the Codex process the
# relay CONNECT proxy; the agent's login shells (Miles, Ray, SGLang, sforge-submit) drop it.
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
