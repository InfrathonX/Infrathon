# Mounted into Work at /etc/profile.d when the task JSON sets "internet": true. HTTP(S)_PROXY is the
# run relay's CONNECT proxy (public hosts on ports 80/443 only), so it stays set for the agent's
# shells; this adds the container's own names and addresses to NO_PROXY so that Miles, Ray and
# SGLang traffic inside Work never enters the proxy. (no-proxy.profile.sh is the internet=false variant.)
_work_direct="localhost,127.0.0.1,::1,$(hostname 2>/dev/null)"
for _work_addr in $(hostname -I 2>/dev/null); do
  _work_direct="${_work_direct},${_work_addr}"
done
case ",${NO_PROXY:-}," in
  *",${_work_direct},"*) ;;
  *) NO_PROXY="${NO_PROXY:+${NO_PROXY},}${_work_direct}" ;;
esac
no_proxy="${NO_PROXY}"
export NO_PROXY no_proxy
unset _work_direct _work_addr
