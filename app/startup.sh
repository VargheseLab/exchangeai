
#!/bin/bash
redis-server --bind 127.0.0.1 --save '' --appendonly no &
redis_pid=$!

celery -A server_init.celery worker --without-mingle --without-gossip &
celery_pid=$!

uwsgi --ini uwsgi.ini &
uwsgi_pid=$!

wait $redis_pid
wait $uwsgi_pid
wait $celery_pid

exit $?