(define (process-cakes pattern)
  (let* ((filelist (cadr (file-glob pattern 1))))
    (while (not (null? filelist))
           (let* ((filename (car filelist))
                  (image (car (gimp-file-load RUN-NONINTERACTIVE
                                              filename filename)))
                  (drawable (car (gimp-image-get-active-layer image))))
             (python-fu-process-NCS-xcf RUN-NONINTERACTIVE image drawable))
           (set! filelist (cdr filelist)))))
