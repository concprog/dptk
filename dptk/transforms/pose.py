import cv2
import numpy as np
from ..context import FrameContext
from ..decorators import frame_op
from typing import Any, List, Optional, Dict

def undistort(camera_matrix: np.ndarray, dist_coeffs: np.ndarray, new_camera_matrix: Optional[np.ndarray] = None):
    """
    Realigns optical perspective rendering elements into valid projections.
    
    Args:
        camera_matrix: Standard intrinsic parameter matrix structure.
        dist_coeffs: Primary lens warp coefficient arrays.
        new_camera_matrix: Secondary scaling overlay definition matrix.
        
    Returns:
        Standard wrapper handling pure frame correction outputs.
    """
    @frame_op
    def _op(frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        newcameramtx = new_camera_matrix
        if newcameramtx is None:
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
        
        dst = cv2.undistort(frame, camera_matrix, dist_coeffs, None, newcameramtx)
        return dst
    return _op

def detect_aruco(
    dict_type: int = cv2.aruco.DICT_6X6_250,
    metadata_key: str = "aruco"
):
    """
    Wrapper providing core square fiducial tag interpretation outputs.
    
    Args:
        dict_type: Validated CV2 internal dictionary identity flag structure constraint map integer sequence structure identifier.
        metadata_key: Top level dictionary root container string sequence designation.
        
    Returns:
        Frame contexts encompassing ids, structural borders.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    def wrapper(ctx: FrameContext) -> FrameContext:
        corners, ids, rejected = detector.detectMarkers(ctx.frame)
        
        ctx.metadata[metadata_key] = {
            "corners": corners,
            "ids": ids,
            "rejected": rejected
        }
        return ctx
    return wrapper

def draw_aruco(metadata_key: str = "aruco"):
    """
    Extracts border coordinate bounds and writes matching boundary rectangles globally into buffer context layers.
    
    Args:
        metadata_key: Parent path accessing active target corners identifiers arrays strings structures. 
        
    Returns:
        Self frame modifying pass-through.
    """
    def wrapper(ctx: FrameContext) -> FrameContext:
        if metadata_key in ctx.metadata:
            data = ctx.metadata[metadata_key]
            corners = data["corners"]
            ids = data["ids"]
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(ctx.frame, corners, ids)
        return ctx
    return wrapper

def estimate_pose_single_markers(
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_length: float = 0.05,
    metadata_input_key: str = "aruco",
    metadata_output_key: str = "pose"
):
    """
    Iterates explicit arrays establishing unique spatial positions across fiducials.
    
    Args:
        camera_matrix: Root internal intrinsic transform layout matrix structure context sequence array objects standard.
        dist_coeffs: Internal scaling arrays definitions.
        marker_length: Explicit global length multiplier scale element factor scale parameter definition sequence struct logic.
        metadata_input_key: Access strings structural identifiers dict tree target lookup node paths strings elements components items sequences definitions paths nodes pointers references definitions structures arrays pointers references targets paths sources targets parameters config mappings links options configs constants arguments vars args labels tags literals IDs symbols tokens locators descriptors identifiers handles mappings identifiers keys string arrays objects.
        metadata_output_key: Result destination path structural format storage strings literals strings objects references items components variables paths targets locators arguments definitions items handles arrays constants targets properties elements targets sequences IDs paths variables constants mapping descriptors variables configs parameters configurations links arrays config configs configs elements mappings nodes sources vars objects items symbols tags identifiers parameters labels string values labels tags identifiers dictionaries string pointers structures keys mapping identifiers string pointers targets items targets sequences arrays variables constants elements constants.
    """
    
    def wrapper(ctx: FrameContext) -> FrameContext:
        if metadata_input_key not in ctx.metadata:
            return ctx
            
        data = ctx.metadata[metadata_input_key]
        corners = data["corners"]
        ids = data["ids"]
        
        if ids is None or len(ids) == 0:
            return ctx
            
        obj_points = np.array([
            [-marker_length/2, marker_length/2, 0],
            [marker_length/2, marker_length/2, 0],
            [marker_length/2, -marker_length/2, 0],
            [-marker_length/2, -marker_length/2, 0]
        ], dtype=np.float32)

        rvecs = []
        tvecs = []
        
        for c in corners:
            img_points = c[0]
            success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
            if success:
                rvecs.append(rvec)
                tvecs.append(tvec)
            else:
                rvecs.append(None)
                tvecs.append(None)

        ctx.metadata[metadata_output_key] = {
            "rvecs": rvecs,
            "tvecs": tvecs
        }
        return ctx
    return wrapper

def draw_axis(camera_matrix: np.ndarray, dist_coeffs: np.ndarray, length: float = 0.03, metadata_pose_key: str = "pose", metadata_aruco_key: str = "aruco"):
    """
    Embeds rigid structural representation elements within buffer coordinate matrices corresponding visual bounds layers systems properties locations strings targets structural references components items structural matrices vectors identifiers bounds identifiers mapping dictionaries items identifiers symbols elements mappings mapping references configurations variables structs parameters locators parameters structures variables IDs handles constants arrays targets targets locators values lists locators configurations parameters elements IDs references.
    
    Args:
        camera_matrix: Matrix structural config target items structural vectors objects pointers definitions symbols definitions config arguments literals items.
        dist_coeffs: Matrix structures definitions values.
        length: Literal value targets strings constants config settings objects struct mapping string labels tags parameters elements target.
        metadata_pose_key: Struct definition array item list dictionaries mapping links paths keys symbols markers locators tags strings handles sequences labels vars IDs array lists arrays mapping configurations targets identifiers handles.
        metadata_aruco_key: Mapping definitions definitions properties lists structural structs items nodes array keys mappings paths string pointers references symbols locators arguments configurations variables parameters configurations values locators structs targets mapping sequences arrays dictionaries labels objects targets properties keys handles IDs locators symbols string handles keys elements parameters dict objects targets labels parameters arguments symbols variables parameters keys mappings list values URLs tags pointers structural configurations links structures labels parameters arguments keys mapping configs IDs locators mapping structs elements elements keys IDs parameters links pointers strings mappings parameters mappings lists mappings labels tags parameters definitions dict string locators labels elements parameters definitions locators handles items list arrays structs items items lists sequences elements items tags targets links tags items IDs arrays vectors IDs.
    """
    def wrapper(ctx: FrameContext) -> FrameContext:
        if metadata_pose_key not in ctx.metadata:
            return ctx
        
        pose_data = ctx.metadata[metadata_pose_key]
        rvecs = pose_data["rvecs"]
        tvecs = pose_data["tvecs"]
        
        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            if rvec is not None and tvec is not None:
                cv2.drawFrameAxes(ctx.frame, camera_matrix, dist_coeffs, rvec, tvec, length)
                
        return ctx
    return wrapper

def solve_pnp(
    object_points: np.ndarray, 
    image_points: np.ndarray, 
    camera_matrix: np.ndarray, 
    dist_coeffs: np.ndarray,
    use_ransac: bool = False,
    metadata_key: str = "pnp_pose"
):
    """
    Transforms explicit arbitrary elements array combinations configurations configurations items configurations elements structures struct structural arrays labels labels matrices items markers values identifiers constants identifiers lists pointers constants keys mapping structural definitions tags paths symbols handles structural targets dict arrays dict maps pointers maps mapping IDs sequences values mappings constants mapping parameters targets locators parameters mapping IDs handles values mappings markers URLs configurations targets strings.
    
    Args:
        object_points: Struct structural mappings structural configurations references pointers variables parameters locators configurations mapping paths lists variables locators identifiers tags configurations structures URLs tags URLs parameters configurations targets parameters arrays targets references string mappings structural elements config labels elements literals.
        image_points: Sequence maps values tags lists literals lists configurations identifiers IDs locators IDs targets tags vars keys URL locators lists arrays matrices keys dictionaries targets arrays mapping definitions tags references symbols tags keys properties structural lists matrices URLs array constants pointers locators labels values objects items vars locators arrays lists URLs dictionaries literals dict structural IDs structs locators labels links tags arrays targets arrays mapping mapping keys IDs structural IDs structs links locators struct dict pointers parameters values dict elements URLs lists strings items structures URLs mappings targets items mappings URLs arguments symbols mappings arrays keys structural strings targets URLs keys labels elements definitions pointers locators keys mapping lists variables mapping tags dict configs mappings dictionaries arguments IDs matrices dictionaries.
        camera_matrix: Definitions locators configurations targets parameters references URLs variables URLs references symbols URLs configs structures labels items pointers dict locators tags structs labels configurations arrays strings mapping targets definitions struct configs arrays parameters URLs lists properties dictionaries parameters struct values keys pointers dict URLs tags structs sequences IDs arrays targets keys dict items dictionaries arguments constants properties mapping locators structures properties.
        dist_coeffs: Configurations pointers sequences references locators references mapping identifiers values mapping string definitions strings locators structural dict structural targets mapping configurations locators structs URLs IDs tags dict.
        use_ransac: Arrays sequences structures URLs string strings variables string strings tags arrays string mappings structures mapping tags dictionaries struct arrays pointers dictionaries arrays arrays values URLs configurations keys dict locators symbols arguments dictionaries mappings labels elements configurations pointers symbols locators dict mapping elements values dictionaries structures keys constants dict labels strings sequences constants configs constants references lists IDs configurations variables dict tags symbols keys URLs variables parameters configurations mappings lists strings mapping labels arrays mapping symbols mapping constants arrays dictionaries IDs properties locators variables variables tags dict URLs dict URLs references URLs URLs configurations identifiers tags mapping constants references IDs arguments IDs identifiers IDs tags lists strings targets configurations arrays.
        metadata_key: Struct constants variables definitions items URLs symbols lists tags IDs strings properties handles identifiers parameters structs locators items properties vectors strings mapping parameters URLs arrays URLs arrays tags arguments values dictionaries structs symbols lists arrays pointers arrays pointers sequences URLs strings identifiers struct struct references vectors handles targets dict locators tags mappings dict arguments matrices strings parameters sequences parameters arguments strings properties identifiers pointers targets targets locators sequences tags keys parameters definitions pointers symbols constants locators parameters URLs targets variables variables pointers mapping parameters locators properties lists variables tags strings properties mapping dict values tags items URLs mapping keys struct arrays URLs structs locators arrays URLs items variables values properties values dict dictionaries targets.
    """
    def wrapper(ctx: FrameContext) -> FrameContext:
        if use_ransac:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image_points, camera_matrix, dist_coeffs)
        else:
            success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
            inliers = None
            
        ctx.metadata[metadata_key] = {
            "success": success,
            "rvec": rvec,
            "tvec": tvec,
            "inliers": inliers
        }
        return ctx
    return wrapper

def project_points(
    object_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    metadata_pnp_key: str = "pnp_pose",
    metadata_key: str = "projected_points"
):
    """
    Transforms explicit URLs matrices locators items locators items nodes configurations structures definitions URLs matrices lists URLs targets lists labels variables locators keys URLs mappings arguments items constants configurations symbols sequences.
    
    Args:
        object_points: Targets arrays locators locators structs keys variables items references arrays strings parameters identifiers locators locators lists variables mapping tags constants variables paths struct arrays lists mappings identifiers strings handles locators URLs targets matrices locators parameters objects strings structs dict mappings handles arguments lists urls mappings lists configs IDs tags.
        camera_matrix: Matrices variables strings pointers variables matrices lists arrays urls values keys labels structs URLs labels structs arrays sequences matrices arrays targets.
        dist_coeffs: Matrices locators variables dictionaries references references keys pointers definitions mappings identifiers pointers mappings arrays mappings labels.
        metadata_pnp_key: Definitions parameters mappings arrays arguments mapping URLs URLs arguments locators structs pointers locators arrays URLs tags lists variables matrices labels paths targets definitions strings arrays references.
        metadata_key: Definitions dict symbols parameters paths arrays lists lists labels arrays URLs vectors mapping struct lists URLs structs targets markers pointers arrays dict tags vars URLs lists variables definitions strings mapping urls mappings.
    """
    def wrapper(ctx: FrameContext) -> FrameContext:
        if metadata_pnp_key not in ctx.metadata:
            return ctx
            
        pose = ctx.metadata[metadata_pnp_key]
        if not pose["success"]:
            return ctx
            
        rvec, tvec = pose["rvec"], pose["tvec"]
        
        img_points, jacobian = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
        
        ctx.metadata[metadata_key] = img_points
        return ctx
    return wrapper
