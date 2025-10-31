from scipy.spatial import distance
import numpy as np

class CentroidTracker:
    """
    A simple object tracker based on object centroids and Euclidean distance.
    It assigns a unique, persistent ID to each object and manages registration
    and deregistration across frames.
    """
    def __init__(self, maxDisappeared=40):
        # Initialize the next unique object ID (will increment for each new object)
        self.nextObjectID = 0
        # Stores the mapping of objectID to its latest centroid [x, y]
        self.objects = {}
        # Stores the number of consecutive frames an object has been 'disappeared'
        self.disappeared = {}
        # Maximum number of frames a person can be "disappeared" before being deregistered
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        """Registers a new object with a unique ID."""
        # Use the next available object ID and store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        """Removes an object from tracking."""
        # Deregister by deleting the object ID from both dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        """
        Updates the tracker with a new set of bounding boxes (rects).
        :param rects: A list of new bounding boxes in the format [startX, startY, endX, endY].
        :return: A dictionary of tracked object IDs and their updated centroids.
        """

        # --- PHASE 1: Handle No Detections ---
        if len(rects) == 0:
            # Go through all existing tracked objects and increment their 'disappeared' counter
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                # Deregister if max disappearance limit is reached
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            # Return the tracked objects
            return self.objects

        # --- PHASE 2: Compute Centroids for New Detections ---
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # Calculate centroid (geometric center of the box)
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # --- PHASE 3: Associate Old and New Centroids ---
        if len(self.objects) == 0:
            # If no objects are currently tracked, register all new detections
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            # Grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # Compute the distance between each pair of existing objects and new input centroids
            # This is an N x M matrix where N is old objects and M is new detections
            D = distance.cdist(np.array(objectCentroids), inputCentroids)

            # Find the smallest value in each row (minimum distance to a new centroid)
            rows = D.min(axis=1).argsort()
            # Find the smallest value in each column (index of the closest existing object)
            cols = D.argmin(axis=1)[rows]

            # Used to keep track of already examined rows and columns
            usedRows = set()
            usedCols = set()
            
            # Loop over the combinations of (row, column) indexes
            for (row, col) in zip(rows, cols):
                # If we've already examined either the row or column value, ignore it
                if row in usedRows or col in usedCols:
                    continue

                # Association: Grab the object ID for the current row, set its new centroid, and reset the disappeared counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                # Note that we've used this row and column
                usedRows.add(row)
                usedCols.add(col)

            # --- PHASE 4: Handle Disappeared and New Objects ---

            # Compute the row and column indices that *have not* been used
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # A. Handle disappeared objects (more old objects than new)
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    # Deregister the object if it has been disappeared for too long
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            
            # B. Handle new objects (more new objects than old)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        # Return the set of tracked objects
        return self.objects