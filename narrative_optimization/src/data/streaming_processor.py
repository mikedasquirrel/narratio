"""
Streaming Data Processor

Processes data streams in real-time for online learning.

Author: Narrative Integration System
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Optional, Callable, Generator
from collections import deque
from datetime import datetime
import threading
import queue


class StreamProcessor:
    """
    Process streaming data in real-time.
    
    Features:
    - Buffered processing
    - Windowing (tumbling, sliding, session)
    - Stream transformations
    - Async processing
    """
    
    def __init__(
        self,
        buffer_size: int = 100,
        window_size: int = 1000,
        window_type: str = 'sliding'
    ):
        self.buffer_size = buffer_size
        self.window_size = window_size
        self.window_type = window_type
        
        self.buffer = []
        self.window = deque(maxlen=window_size if window_type == 'sliding' else None)
        
        self.total_processed = 0
        self.last_processed_time = None
        
    def ingest(self, item: Dict):
        """
        Ingest single item into stream.
        
        Parameters
        ----------
        item : dict
            Item to ingest
        """
        item['_ingested_at'] = datetime.now().timestamp()
        
        self.buffer.append(item)
        self.window.append(item)
        
        # Process buffer if full
        if len(self.buffer) >= self.buffer_size:
            self.process_buffer()
        
        self.total_processed += 1
        self.last_processed_time = datetime.now()
    
    def process_buffer(self):
        """Process buffered items."""
        if len(self.buffer) == 0:
            return
        
        # Extract texts and outcomes
        texts = [item.get('text', '') for item in self.buffer]
        outcomes = np.array([item.get('outcome', 0) for item in self.buffer])
        
        # Trigger processing callbacks
        for callback in getattr(self, '_callbacks', []):
            callback(texts, outcomes)
        
        # Clear buffer
        self.buffer = []
    
    def register_callback(self, callback: Callable):
        """
        Register callback for buffer processing.
        
        Parameters
        ----------
        callback : callable
            Function to call with (texts, outcomes)
        """
        if not hasattr(self, '_callbacks'):
            self._callbacks = []
        self._callbacks.append(callback)
    
    def get_window_snapshot(self) -> List[Dict]:
        """Get current window contents."""
        return list(self.window)
    
    def tumbling_windows(
        self,
        stream: Generator[Dict, None, None]
    ) -> Generator[List[Dict], None, None]:
        """
        Tumbling window: non-overlapping fixed-size windows.
        
        Parameters
        ----------
        stream : generator
            Input stream
        
        Yields
        ------
        list
            Window contents
        """
        window_buffer = []
        
        for item in stream:
            window_buffer.append(item)
            
            if len(window_buffer) >= self.window_size:
                yield window_buffer
                window_buffer = []
        
        # Final partial window
        if len(window_buffer) > 0:
            yield window_buffer
    
    def sliding_windows(
        self,
        stream: Generator[Dict, None, None],
        slide_size: int = 100
    ) -> Generator[List[Dict], None, None]:
        """
        Sliding window: overlapping windows.
        
        Parameters
        ----------
        stream : generator
            Input stream
        slide_size : int
            How much to slide
        
        Yields
        ------
        list
            Window contents
        """
        window_buffer = deque(maxlen=self.window_size)
        count = 0
        
        for item in stream:
            window_buffer.append(item)
            count += 1
            
            # Emit window every slide_size items
            if count >= self.window_size and count % slide_size == 0:
                yield list(window_buffer)
    
    def session_windows(
        self,
        stream: Generator[Dict, None, None],
        session_gap: float = 300.0
    ) -> Generator[List[Dict], None, None]:
        """
        Session window: gap-based windows.
        
        Parameters
        ----------
        stream : generator
            Input stream
        session_gap : float
            Max gap (seconds) within session
        
        Yields
        ------
        list
            Session contents
        """
        session = []
        last_time = None
        
        for item in stream:
            current_time = item.get('_ingested_at', datetime.now().timestamp())
            
            if last_time is not None:
                gap = current_time - last_time
                
                if gap > session_gap:
                    # Gap exceeded - emit session and start new
                    if len(session) > 0:
                        yield session
                    session = []
            
            session.append(item)
            last_time = current_time
        
        # Final session
        if len(session) > 0:
            yield session
    
    def transform_stream(
        self,
        stream: Generator[Dict, None, None],
        transform_func: Callable[[Dict], Dict]
    ) -> Generator[Dict, None, None]:
        """
        Transform each item in stream.
        
        Parameters
        ----------
        stream : generator
            Input stream
        transform_func : callable
            Transformation function
        
        Yields
        ------
        dict
            Transformed items
        """
        for item in stream:
            yield transform_func(item)
    
    def filter_stream(
        self,
        stream: Generator[Dict, None, None],
        predicate: Callable[[Dict], bool]
    ) -> Generator[Dict, None, None]:
        """
        Filter stream items.
        
        Parameters
        ----------
        stream : generator
            Input stream
        predicate : callable
            Filter predicate
        
        Yields
        ------
        dict
            Filtered items
        """
        for item in stream:
            if predicate(item):
                yield item
    
    def get_stats(self) -> Dict:
        """Get processing statistics."""
        return {
            'total_processed': self.total_processed,
            'buffer_size': len(self.buffer),
            'window_size': len(self.window),
            'last_processed': self.last_processed_time.isoformat() if self.last_processed_time else None
        }


class AsyncStreamProcessor(StreamProcessor):
    """
    Async stream processor using threading.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.input_queue = queue.Queue()
        self.is_running = False
        self.worker_thread = None
        
    def start(self):
        """Start async processing."""
        if self.is_running:
            return
        
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
    
    def stop(self):
        """Stop async processing."""
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join()
    
    def _worker(self):
        """Worker thread for processing."""
        while self.is_running:
            try:
                item = self.input_queue.get(timeout=1.0)
                super().ingest(item)
            except queue.Empty:
                continue
    
    def ingest_async(self, item: Dict):
        """Ingest item asynchronously."""
        self.input_queue.put(item)

