
import asyncio
from functools import wraps
import time
from urllib.parse import urlparse

from flask import request, session

def limit_content_length():
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            cl = request.content_length
            cfp = request.max_form_parts
            request.__setattr__('max_form_parts', 200000)
            return f(*args, **kwargs)
        return wrapper
    return decorator

def with_referrer():
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            parsed_url = urlparse(request.referrer)
            path_parts = parsed_url.path.split('/')
            if path_parts[1] == 'finetune' and len(path_parts) > 2:
                referrer = f"{path_parts[1]}/{path_parts[2]}"
            else:
                referrer = f"{path_parts[-1]}"
            return f(*args, **kwargs, referrer=referrer)
        return wrapper
    return decorator

    

def with_session_key():
    def decorator(f):
        @wraps(f)
        async def async_wrapper(*args, **kwargs):
            if 'session_key' in kwargs and kwargs['session_key'] is not None:
                #session_value = kwargs['session_key']
                return f(*args, **kwargs)
            else:
                session_value = session.sid
                kwargs.pop('session_key', None)
                return f(*args, **kwargs, session_key=session_value)
            #return await f(*args, **kwargs, session_key=session_value)

        @wraps(f)
        def sync_wrapper(*args, **kwargs):
            if 'session_key' in kwargs and kwargs['session_key'] is not None:
                #session_value = kwargs['session_key']
                return f(*args, **kwargs)
            else:
                session_value = session.sid
                kwargs.pop('session_key', None)
                return f(*args, **kwargs, session_key=session_value)

        # Check if the function is a coroutine
        if asyncio.iscoroutinefunction(f):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator
        
def time_cache(seconds: int):
    def decorator(func):
        cache = {}
        
        @wraps(func)
        def wrapped(*args, **kwargs):
            nonlocal cache
            now = time.time()
            
            # Clean up expired cache entries
            cache = {k: v for k, v in cache.items() if now - v[1] < seconds}
            
            key = (args, tuple(kwargs.items()))
            if key in cache:
                return cache[key][0]
            
            # Call the function and store the result with the current timestamp
            result = func(*args, **kwargs)
            cache[key] = (result, now)
            return result
        
        return wrapped
    return decorator