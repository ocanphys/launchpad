APP_NAME = "launchpad"
VOLUME_NAME = "trainvols"
STORAGE = "/storage"  # this is the container mount name for the volume.
CONTAINER_LIFETIME = 3600  # no container lives beyond this many seconds.
HEARTBEAT_SECONDS = 1
FLATLINE = 5  # if heartbeat age is longer than this many HEARTBEAT_SECONDS, the call is not active.
