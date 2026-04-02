"""
3D Model to 2D CAD Drawing Converter
Uses Trimesh to convert 3D models into 2D orthographic projections
Supports multiple output formats: DXF, SVG, and PNG
"""

import trimesh
import numpy as np
from pathlib import Path
import argparse
import sys
import time
import os


class Model3Dto2DConverter:
    """Convert 3D models to 2D CAD drawings with orthographic projections"""
    
    def __init__(self, model_path):
        """
        Initialize converter with a 3D model file
        
        Args:
            model_path: Path to 3D model file (STL, OBJ, PLY, STEP/STP, etc.)
        """
        self.model_path = Path(model_path)
        self.mesh = None
        self._load_start = time.time()
        self._cpu_start = time.process_time()
        self.load_model()
        self._load_time = time.time() - self._load_start
        print(f"  ⏱  Model load time: {self._load_time:.3f}s")
        
    def load_model(self):
        """Load the 3D model using Trimesh"""
        try:
            file_type = None
            if self.model_path.suffix.lower() in {'.stp', '.step'}:
                file_type = 'step'
            
            # Try loading with trimesh first
            try:
                loaded = trimesh.load(str(self.model_path), file_type=file_type)
            except Exception as e:
                # If trimesh fails with STEP files, try cadquery
                if self.model_path.suffix.lower() in {'.stp', '.step'}:
                    print(f"  Trimesh failed, trying CadQuery for STEP file...")
                    loaded = self._load_step_with_cadquery()
                else:
                    raise e
            
            # Handle Scene objects (GLTF, OBJ with multiple objects, etc.)
            if isinstance(loaded, trimesh.Scene):
                print(f"[+] Loaded scene: {self.model_path.name}")
                print(f"  Scene contains {len(loaded.geometry)} geometries")
                
                # Combine all geometries into a single mesh
                meshes = []
                for name, geom in loaded.geometry.items():
                    if isinstance(geom, trimesh.Trimesh):
                        meshes.append(geom)
                
                if not meshes:
                    raise ValueError("No valid meshes found in scene")
                
                # Concatenate all meshes
                if len(meshes) == 1:
                    self.mesh = meshes[0]
                else:
                    self.mesh = trimesh.util.concatenate(meshes)
                    print(f"  Combined {len(meshes)} meshes into one")
            else:
                # Single mesh object
                self.mesh = loaded
            
            # Adaptive subdivision for better accuracy on curved surfaces
            # Analyze mesh complexity and apply intelligent subdivision
            complexity_score = self._analyze_mesh_complexity(self.mesh)
            
            # Determine subdivision strategy based on complexity
            if complexity_score['needs_subdivision']:
                try:
                    original_faces = len(self.mesh.faces)
                    
                    # Apply adaptive subdivision based on complexity
                    self.mesh = self._adaptive_subdivision(
                        self.mesh, 
                        max_faces=80000,  # Reduced from 100k for better performance
                        complexity_score=complexity_score
                    )
                    
                    print(f"  Adaptively subdivided mesh: {original_faces} → {len(self.mesh.faces)} faces")
                    print(f"  Complexity score: {complexity_score['score']:.2f} (curvature variance: {complexity_score['curvature_variance']:.4f})")
                except Exception as e:
                    print(f"  Note: Adaptive subdivision skipped: {e}")
            
            print(f"[+] Loaded model: {self.model_path.name}")
            print(f"  Vertices: {len(self.mesh.vertices)}")
            print(f"  Faces: {len(self.mesh.faces)}")
            print(f"  Bounds: {self.mesh.bounds}")
        except Exception as e:
            print(f"[!] Error loading model: {e}")
            if self.model_path.suffix.lower() in {'.stp', '.step'}:
                print("  Hint: STEP/STP files require cadquery or pythonocc-core.")
                print("  Install with: pip install cadquery")
            sys.exit(1)
    
    def _load_step_with_cadquery(self):
        """Load STEP file using CadQuery and convert to trimesh"""
        try:
            import cadquery as cq
            import tempfile
            
            # Load STEP file with cadquery
            print(f"  Loading STEP file with CadQuery...")
            result = cq.importers.importStep(str(self.model_path))
            
            # Export to STL in a temporary file
            with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp:
                tmp_path = tmp.name
            
            try:
                # Export the shape to STL with high quality settings
                # Lower tolerance = more detail (0.01 is 10x finer than default 0.1)
                cq.exporters.export(result, tmp_path, exportType='STL', tolerance=0.01, angularTolerance=0.1)
                
                # Load the STL with trimesh
                mesh = trimesh.load(tmp_path)
                
                print(f"  [+] Loaded STEP file via CadQuery")
                
                return mesh
            finally:
                # Clean up temporary file
                import os
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            
        except ImportError:
            raise ImportError("CadQuery not installed. Install with: pip install cadquery")
        except Exception as e:
            raise Exception(f"Failed to load STEP file with CadQuery: {e}")
    
    def _analyze_mesh_complexity(self, mesh):
        """
        Analyze mesh complexity to determine optimal subdivision and processing parameters
        
        Returns:
            dict with complexity metrics and recommendations
        """
        try:
            # Calculate basic metrics
            num_faces = len(mesh.faces)
            num_vertices = len(mesh.vertices)
            
            # Calculate edge lengths for detail analysis
            edges = mesh.edges_unique
            edge_vectors = mesh.vertices[edges[:, 1]] - mesh.vertices[edges[:, 0]]
            edge_lengths = np.linalg.norm(edge_vectors, axis=1)
            
            # Analyze edge length distribution
            mean_edge_length = np.mean(edge_lengths)
            std_edge_length = np.std(edge_lengths)
            edge_length_variance = std_edge_length / (mean_edge_length + 1e-10)
            
            # Analyze curvature using face normals
            face_normals = mesh.face_normals
            
            # Calculate normal variation (proxy for curvature)
            # Compare each face normal with its neighbors
            face_adjacency = mesh.face_adjacency
            if len(face_adjacency) > 0:
                normal_diffs = []
                for adj in face_adjacency:
                    n1 = face_normals[adj[0]]
                    n2 = face_normals[adj[1]]
                    # Angle between normals (0 = flat, π = sharp edge)
                    dot_product = np.clip(np.dot(n1, n2), -1.0, 1.0)
                    angle = np.arccos(dot_product)
                    normal_diffs.append(angle)
                
                curvature_variance = np.std(normal_diffs)
                mean_curvature = np.mean(normal_diffs)
            else:
                curvature_variance = 0.0
                mean_curvature = 0.0
            
            # Calculate complexity score (0-1 scale)
            # Higher score = more complex geometry requiring more subdivision
            complexity_factors = [
                min(1.0, edge_length_variance / 0.5),  # Edge length variation
                min(1.0, curvature_variance / 0.3),     # Curvature variation
                min(1.0, mean_curvature / 0.5),         # Average curvature
                min(1.0, num_faces / 50000)             # Model size factor
            ]
            
            complexity_score = np.mean(complexity_factors)
            
            # Smarter subdivision strategy - balance accuracy with performance
            # Only subdivide if mesh is coarse AND has complexity
            needs_subdivision = (
                num_faces < 50000 and  # Only subdivide smaller meshes (reduced from 100k)
                (complexity_score > 0.4 or  # Moderate complexity (increased from 0.3)
                 curvature_variance > 0.2 or  # High curvature variation (increased from 0.15)
                 num_faces < 3000)  # Very coarse mesh (reduced from 5000)
            )
            
            # Limit to single subdivision pass for better performance
            # Only do 2 passes for extremely coarse meshes
            recommended_subdivisions = 2 if (num_faces < 2000 and complexity_score > 0.7) else 1
            
            return {
                'score': complexity_score,
                'num_faces': num_faces,
                'num_vertices': num_vertices,
                'edge_length_variance': edge_length_variance,
                'curvature_variance': curvature_variance,
                'mean_curvature': mean_curvature,
                'needs_subdivision': needs_subdivision,
                'recommended_subdivisions': recommended_subdivisions
            }
            
        except Exception as e:
            print(f"  Warning: Complexity analysis failed: {e}")
            # Return safe defaults
            return {
                'score': 0.5,
                'num_faces': len(mesh.faces),
                'num_vertices': len(mesh.vertices),
                'edge_length_variance': 0.0,
                'curvature_variance': 0.0,
                'mean_curvature': 0.0,
                'needs_subdivision': len(mesh.faces) < 50000,
                'recommended_subdivisions': 1
            }
    
    def _adaptive_subdivision(self, mesh, max_faces=80000, complexity_score=None):
        """
        Adaptively subdivide mesh based on local curvature and complexity
        
        Args:
            mesh: Input mesh
            max_faces: Maximum number of faces after subdivision (reduced from 100k for performance)
            complexity_score: Pre-computed complexity metrics
            
        Returns:
            Subdivided mesh
        """
        try:
            if complexity_score is None:
                complexity_score = self._analyze_mesh_complexity(mesh)
            
            num_subdivisions = complexity_score['recommended_subdivisions']
            current_mesh = mesh.copy()
            
            # Safety check: don't subdivide if it would create too many faces
            estimated_faces = len(current_mesh.faces) * (4 ** num_subdivisions)
            if estimated_faces > max_faces:
                print(f"  Skipping subdivision: would create {estimated_faces} faces (max: {max_faces})")
                return mesh
            
            # Progressive subdivision with decreasing aggressiveness
            for i in range(num_subdivisions):
                if len(current_mesh.faces) >= max_faces:
                    break
                
                # Standard subdivision (quadruples face count)
                current_mesh = current_mesh.subdivide()
                
                # If we're over the limit after subdivision, return original
                if len(current_mesh.faces) > max_faces:
                    print(f"  Subdivision exceeded limit ({len(current_mesh.faces)} > {max_faces}), using original mesh")
                    return mesh
            
            return current_mesh
            
        except Exception as e:
            print(f"  Warning: Adaptive subdivision failed: {e}")
            # Fallback to simple subdivision only for very coarse meshes
            try:
                if len(mesh.faces) < max_faces // 4:
                    return mesh.subdivide()
            except:
                pass
            return mesh
    
    def _detect_feature_points(self, mesh, view_direction):
        """
        Detect important feature points (corners, ridges, valleys)
        
        Args:
            mesh: Input mesh
            view_direction: View direction vector
            
        Returns:
            dict with feature points and their importance scores
        """
        try:
            vertices = mesh.vertices
            face_normals = mesh.face_normals
            
            # Find vertices with high curvature (corners, ridges)
            vertex_normals = mesh.vertex_normals
            
            # Calculate curvature at each vertex by analyzing normal variation
            feature_scores = np.zeros(len(vertices))
            
            # Get vertex faces for neighborhood analysis
            vertex_faces = mesh.vertex_faces
            
            for i, faces_idx in enumerate(vertex_faces):
                # Filter out invalid face indices
                valid_faces = faces_idx[faces_idx != -1]
                
                if len(valid_faces) < 2:
                    continue
                
                # Get normals of adjacent faces
                adjacent_normals = face_normals[valid_faces]
                
                # Calculate normal variation (higher = sharper feature)
                normal_variance = np.std([np.dot(vertex_normals[i], n) for n in adjacent_normals])
                
                # Calculate visibility from view direction
                visibility = abs(np.dot(vertex_normals[i], view_direction))
                
                # Combined feature score
                feature_scores[i] = normal_variance * visibility
            
            # Identify top feature points
            threshold = np.percentile(feature_scores, 85)  # Top 15% of features
            feature_indices = np.where(feature_scores > threshold)[0]
            
            return {
                'indices': feature_indices,
                'scores': feature_scores[feature_indices],
                'positions': vertices[feature_indices]
            }
            
        except Exception as e:
            print(f"  Warning: Feature detection failed: {e}")
            return {'indices': np.array([]), 'scores': np.array([]), 'positions': np.array([])}
    
    def _calculate_edge_importance(self, edge_props, mesh, view_dir):
        """
        Calculate importance score for an edge based on multiple criteria
        
        Args:
            edge_props: Dictionary with edge properties
            mesh: Input mesh
            view_dir: View direction vector
            
        Returns:
            Importance score (0-1)
        """
        try:
            # Criteria weights
            angle_weight = 0.3
            visibility_weight = 0.25
            length_weight = 0.15
            curvature_weight = 0.3
            
            # Normalize angle (0-π → 0-1)
            angle_score = edge_props['angle'] / np.pi
            
            # Visibility score (how visible is this edge)
            visibility_score = max(abs(edge_props['dot1']), abs(edge_props['dot2']))
            
            # Length score (longer edges are more important)
            # Normalize by mesh scale
            mesh_scale = np.max(mesh.extents)
            length_score = min(1.0, edge_props['length'] / (mesh_scale * 0.1))
            
            # Curvature score (difference in face orientations)
            curvature_score = abs(edge_props['dot1'] - edge_props['dot2'])
            
            # Combined weighted score
            importance = (
                angle_weight * angle_score +
                visibility_weight * visibility_score +
                length_weight * length_score +
                curvature_weight * curvature_score
            )
            
            return min(1.0, importance)
            
        except Exception as e:
            return 0.5  # Default medium importance
    
    def get_projection(self, view='front'):
        """
        Get 2D projection of the mesh from specified view with enhanced accuracy
        
        Args:
            view: 'front', 'top', 'side', 'isometric', or custom view parameters
            
        Returns:
            Path2D or Path3D object containing the projected outline
        """
        # Enhanced accuracy view definitions - ultra-aggressive feature detection
        # Standard 6 orthographic views + isometric
        view_params = {
            'front': {  # Looking along +Z axis (XY plane)
                'rotation': np.eye(4),
                'edge_angle': 5.0,  # More sensitive (was 10.0)
                'feature_scale': 2.5,  # Increased (was 1.5)
                'silhouette_threshold': 0.002,  # Much more sensitive (was 0.01)
                'min_edge_length': 0.00005  # Preserve tiny details (was 0.0005)
            },
            'back': {  # Looking along -Z axis
                'rotation': trimesh.transformations.rotation_matrix(
                    np.radians(180), [0, 1, 0]
                ),
                'edge_angle': 5.0,
                'feature_scale': 2.5,
                'silhouette_threshold': 0.002,
                'min_edge_length': 0.00005
            },
            'top': {  # Looking down along -Y axis (XZ plane)
                'rotation': trimesh.transformations.rotation_matrix(
                    np.radians(90), [1, 0, 0]
                ),
                'edge_angle': 3.0,  # Ultra-sensitive (was 1.0)
                'feature_scale': 3.0,  # Maximum boost (was 2.0)
                'silhouette_threshold': 0.001,  # Extremely sensitive (was 0.005)
                'min_edge_length': 0.00003  # Keep tiniest details (was 0.0001)
            },
            'bottom': {  # Looking up along +Y axis
                'rotation': trimesh.transformations.rotation_matrix(
                    np.radians(-90), [1, 0, 0]
                ),
                'edge_angle': 5.0,
                'feature_scale': 2.5,
                'silhouette_threshold': 0.002,
                'min_edge_length': 0.00005
            },
            'right': {  # Looking along +X axis (YZ plane)
                # Rotate 90 degrees around Z to make it horizontal like the reference image
                'rotation': trimesh.transformations.rotation_matrix(
                    np.radians(90), [0, 1, 0]
                ) @ trimesh.transformations.rotation_matrix(
                    np.radians(-90), [0, 0, 1]
                ),
                'edge_angle': 3.0,  # Ultra-sensitive (was 1.0)
                'feature_scale': 3.0,  # Maximum boost (was 2.0)
                'silhouette_threshold': 0.001,  # Extremely sensitive (was 0.005)
                'min_edge_length': 0.00003  # Keep tiniest details (was 0.0001)
            },
            'left': {  # Looking along -X axis
                'rotation': trimesh.transformations.rotation_matrix(
                    np.radians(-90), [0, 1, 0]
                ),
                'edge_angle': 5.0,
                'feature_scale': 2.5,
                'silhouette_threshold': 0.002,
                'min_edge_length': 0.00005
            },
            'side': {  # Alias for 'right' for backward compatibility
                'rotation': trimesh.transformations.rotation_matrix(
                    np.radians(90), [0, 1, 0]
                ),
                'edge_angle': 5.0,
                'feature_scale': 2.5,
                'silhouette_threshold': 0.002,
                'min_edge_length': 0.00005
            },
            'isometric': {
                'rotation': trimesh.transformations.rotation_matrix(
                    np.radians(35.264), [1, 0, 0]
                ) @ trimesh.transformations.rotation_matrix(
                    np.radians(45), [0, 0, 1]
                ),
                'edge_angle': 4.0,  # More sensitive (was 8.0)
                'feature_scale': 2.8,  # Increased (was 1.6)
                'silhouette_threshold': 0.003,  # More sensitive (was 0.008)
                'min_edge_length': 0.00008  # Preserve more detail (was 0.0003)
            },
            # Section views - show internal features with center cuts
            'section_front': {
                'rotation': np.eye(4),
                'edge_angle': 5.0,
                'feature_scale': 2.5,
                'silhouette_threshold': 0.002,
                'min_edge_length': 0.00005,
                'is_section': True
            },
            'section_top': {
                'rotation': trimesh.transformations.rotation_matrix(
                    np.radians(90), [1, 0, 0]
                ),
                'edge_angle': 3.0,
                'feature_scale': 3.0,
                'silhouette_threshold': 0.001,
                'min_edge_length': 0.00003,
                'is_section': True
            },
            'section_side': {
                'rotation': trimesh.transformations.rotation_matrix(
                    np.radians(90), [0, 1, 0]
                ) @ trimesh.transformations.rotation_matrix(
                    np.radians(-90), [0, 0, 1]
                ),
                'edge_angle': 3.0,
                'feature_scale': 3.0,
                'silhouette_threshold': 0.001,
                'min_edge_length': 0.00003,
                'is_section': True
            }
        }
        
        # Create a copy of the mesh and apply rotation
        mesh_copy = self.mesh.copy()
        
        # Get view parameters or use ultra-accurate defaults
        params = view_params.get(view, {
            'rotation': np.eye(4),
            'edge_angle': 10.0,
            'feature_scale': 1.5,
            'silhouette_threshold': 0.01,
            'min_edge_length': 0.0005
        })
        
        # Apply rotation and prepare mesh
        mesh_copy.apply_transform(params['rotation'])
        
        # Check if this is a section view
        is_section_view = params.get('is_section', False)
        
        if is_section_view:
            # Create section view with cuts and hatching
            # Extract the base view name (remove 'section_' prefix)
            base_view = view.replace('section_', '') if view.startswith('section_') else view
            return self._create_section_view(mesh_copy, base_view)
        
        # Get the 2D outline with enhanced projection
        try:
            # First try with rtree for better accuracy
            planar = mesh_copy.projected(normal=[0, 0, 1])
            
            # If successful, enhance with additional features
            if planar is not None:
                # Extract silhouette edges with view-specific parameters
                enhanced_result = self._extract_silhouette_edges(
                    mesh_copy,
                    edge_angle=params['edge_angle'],
                    feature_scale=params['feature_scale'],
                    silhouette_threshold=params['silhouette_threshold'],
                    min_edge_length=params.get('min_edge_length', 0.001)
                )
                
                # Add section cuts for internal features
                section_result = self._add_section_cuts(mesh_copy, view)
                
                # Combine results
                if section_result and enhanced_result:
                    return self._combine_projections(enhanced_result, section_result)
                return enhanced_result or section_result or planar
                
        except ImportError:
            print("  Note: Install 'rtree' for better projections (pip install rtree)")
        except Exception as e:
            print(f"  Warning: Projection method failed: {e}")
        
        try:
            # Fallback to advanced silhouette extraction
            silhouette_result = self._extract_silhouette_edges(
                mesh_copy,
                edge_angle=params['edge_angle'],
                feature_scale=params['feature_scale'],
                silhouette_threshold=params['silhouette_threshold'],
                min_edge_length=params.get('min_edge_length', 0.001)
            )
            
            # Add section cuts if requested
            section_result = self._add_section_cuts(mesh_copy, view)
            
            # Combine results
            if section_result and silhouette_result:
                return self._combine_projections(silhouette_result, section_result)
            return silhouette_result or section_result or planar
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  Warning: Advanced silhouette extraction failed: {e}")
            # Fallback to alpha shape if all other methods fail
            # Need to project vertices to 2D for alpha shape if not already done
            vertices_2d = mesh_copy.vertices[:, :2]
            return self._create_alpha_shape_projection(vertices_2d)
    
    def _extract_silhouette_edges(self, mesh, edge_angle=15.0, feature_scale=1.2, 
                                   silhouette_threshold=0.02, min_edge_length=0.001):
        """
        Extract silhouette and feature edges with enhanced accuracy
        
        Args:
            mesh: Input mesh
            edge_angle: Minimum angle (degrees) between faces to be considered a feature edge
            feature_scale: Scale factor for feature edge detection sensitivity
            silhouette_threshold: Threshold for silhouette edge detection (0-1)
            min_edge_length: Minimum edge length to preserve (for small features)
            
        Returns:
            Path2D containing the extracted edges
        """
        try:
            # Get all edges and their adjacent faces
            edges = np.asarray(mesh.edges_unique)
            edges_face = np.asarray(mesh.edges_face)
            
            # Project vertices to 2D for final output
            vertices_2d = mesh.vertices[:, :2]
            
            # Enhanced edge detection with multiple criteria
            silhouette_edges = []
            feature_edges = []
            crease_edges = []
            hidden_edges = []  # For hidden line detection
            
            # View direction is along Z axis (0, 0, 1)
            view_dir = np.array([0, 0, 1])
            
            # Pre-compute edge properties
            edge_props = []
            
            # Ensure we have valid arrays to iterate
            if edges.ndim != 2 or edges.shape[1] != 2:
                print(f"  Warning: Invalid edges shape: {edges.shape}")
                return self._create_alpha_shape_projection(vertices_2d)
                
            for i in range(len(edges)):
                try:
                    edge = edges[i]
                    faces = edges_face[i]
                    
                    # Comprehensive edge validation
                    # Check for scalars or invalid shapes
                    if np.ndim(edge) != 1 or len(edge) != 2:
                        continue
                    if np.ndim(faces) != 1: # faces should be (2,)
                        continue
                        
                    # Ensure edge indices are valid integers
                    edge_idx = edge.astype(int)
                    if edge_idx[0] < 0 or edge_idx[1] < 0:
                        continue
                    if edge_idx[0] >= len(mesh.vertices) or edge_idx[1] >= len(mesh.vertices):
                        continue
                    
                    # Boundary edges (only one face) - always include with high priority
                    # faces[1] == -1 check
                    if faces[1] == -1:
                        silhouette_edges.append(edge_idx)
                        continue
                    
                    # For edges with two faces, compute properties
                    if faces[0] != -1 and faces[1] != -1:
                        # Validate face indices
                        if faces[0] >= len(mesh.face_normals) or faces[1] >= len(mesh.face_normals):
                            continue
                        
                        # Get face normals and centers
                        normal1 = mesh.face_normals[faces[0]]
                        normal2 = mesh.face_normals[faces[1]]
                        
                        # Compute angle between faces
                        cos_angle = np.clip(np.dot(normal1, normal2), -1.0, 1.0)
                        angle = np.arccos(cos_angle)
                        
                        # Compute visibility of each face
                        dot1 = np.dot(normal1, view_dir)
                        dot2 = np.dot(normal2, view_dir)
                        
                        # Compute edge midpoint and length - use validated indices
                        v1 = mesh.vertices[edge_idx[0]]
                        v2 = mesh.vertices[edge_idx[1]]
                        edge_center = (v1 + v2) * 0.5
                        edge_length = np.linalg.norm(v2 - v1)
                        
                        # Skip very small edges unless they're important features
                        if edge_length < min_edge_length * 0.1:
                            continue
                        
                        # Store properties for multi-criteria evaluation
                        edge_props.append({
                            'edge': edge_idx,  # Use validated integer indices
                            'angle': angle,
                            'dot1': dot1,
                            'dot2': dot2,
                            'center': edge_center,
                            'length': edge_length,
                            'normal1': normal1,
                            'normal2': normal2
                        })
                except Exception as e:
                    # Skip specific malformed edge but continue with others
                    continue
            
            # Multi-pass edge classification
            for prop in edge_props:
                # Silhouette edge detection (one face visible, one hidden)
                is_silhouette = False
                dot1, dot2 = prop['dot1'], prop['dot2']
                
                # Check for silhouette condition with threshold
                if (dot1 > silhouette_threshold and dot2 < -silhouette_threshold) or \
                   (dot1 < -silhouette_threshold and dot2 > silhouette_threshold):
                    silhouette_edges.append(prop['edge'])
                    is_silhouette = True
                
                # Feature edge detection (sharp angles between faces) - ultra-aggressive
                if not is_silhouette and prop['angle'] > np.radians(edge_angle):
                    # Weight by visibility and edge length
                    visibility = max(abs(dot1), abs(dot2))
                    weight = visibility * prop['length'] * feature_scale
                    
                    # Ultra-low threshold for maximum feature detection
                    if weight > 0.01:  # Even more sensitive (was 0.02)
                        if abs(dot1 - dot2) > 0.2:  # More lenient (was 0.3)
                            silhouette_edges.append(prop['edge'])
                        else:
                            feature_edges.append(prop['edge'])
                
                # Detect crease edges (curvature-based) with ultra-low threshold
                if not is_silhouette and prop['angle'] > np.radians(edge_angle * 0.2):  # Even lower (was 0.3)
                    # Check if this is a crease (both faces facing similar direction)
                    # Increased tolerance to capture more curvature
                    if dot1 * dot2 > 0 and abs(dot1 - dot2) < 0.7:  # More lenient (was 0.6)
                        crease_edges.append(prop['edge'])
                
                # Hidden edge detection (both faces hidden but edge is important)
                if not is_silhouette and dot1 < -0.05 and dot2 < -0.05:  # More sensitive (was -0.1)
                    # Both faces facing away, but sharp edge
                    if prop['angle'] > np.radians(edge_angle * 1.2):  # Lower threshold (was 1.5)
                        hidden_edges.append(prop['edge'])
            
            # Combine all edge types with priority (silhouette > feature > crease > hidden)
            all_edges = silhouette_edges + feature_edges + crease_edges + hidden_edges
            
            # Store edge types for later rendering (e.g., dashed lines for hidden)
            edge_types = {}
            # Helper to safely convert to tuple of python ints
            def to_edge_key(e):
                try:
                    return tuple(map(int, e))
                except:
                    return (0, 0)

            for edge in silhouette_edges:
                edge_types[to_edge_key(edge)] = 'silhouette'
            for edge in feature_edges:
                edge_types[to_edge_key(edge)] = 'feature'
            for edge in crease_edges:
                edge_types[to_edge_key(edge)] = 'crease'
            for edge in hidden_edges:
                edge_types[to_edge_key(edge)] = 'hidden'
            
            # If no edges found, fall back to alpha shape
            if not all_edges:
                return self._create_alpha_shape_projection(mesh.vertices[:, :2])
            
            # Create vertex and edge mapping with deduplication
            unique_verts = []
            vert_map = {}
            edge_list = []
            
            # Process edges with priority to silhouette edges
            for edge in all_edges:
                # Skip invalid edges
                if len(edge) != 2:
                    continue
                    
                # Map vertices to indices
                try:
                    v1, v2 = int(edge[0]), int(edge[1])
                    if v1 not in vert_map:
                        vert_map[v1] = len(unique_verts)
                        unique_verts.append(vertices_2d[v1])
                    if v2 not in vert_map:
                        vert_map[v2] = len(unique_verts)
                        unique_verts.append(vertices_2d[v2])
                    
                    # Add edge with new indices
                    edge_list.append([vert_map[v1], vert_map[v2]])
                except (IndexError, ValueError, TypeError):
                    continue
            
            # Create path entities with proper connectivity
            entities = []
            for edge in edge_list:
                # Skip invalid edges
                if len(edge) != 2 or edge[0] == edge[1]:
                    continue
                entities.append(trimesh.path.entities.Line(points=edge))
            
            # Create final path
            if not entities or len(unique_verts) < 2:
                return self._create_alpha_shape_projection(mesh.vertices[:, :2])
                
            path = trimesh.path.Path2D(
                entities=entities,
                vertices=np.array(unique_verts)
            )
            
            # Advanced multi-pass refinement for cleaner results
            try:
                # First pass: merge very close vertices (tighter tolerance)
                path.merge_vertices(merge_tex=True, merge_norm=True)
                
                # Second pass: remove duplicate entities
                path.remove_duplicate_entities()
                
                # Third pass: adaptive simplification based on entity count
                # More aggressive simplification for very dense paths
                if len(path.entities) > 500:
                    try:
                        # Stronger simplification for very dense paths
                        path = path.simplify_spline(smooth=0.0005)
                    except:
                        pass
                elif len(path.entities) > 100:
                    try:
                        # Gentle simplification to remove noise while preserving features
                        path = path.simplify_spline(smooth=0.0001)
                    except:
                        pass  # Simplification not always available
                
                # Fourth pass: merge again after simplification
                try:
                    path.merge_vertices(merge_tex=True, merge_norm=True)
                except:
                    pass
                        
            except Exception as e:
                print(f"  Note: Path optimization failed: {e}")
            
            # Store edge type metadata if available
            if hasattr(path, 'metadata'):
                path.metadata['edge_types'] = edge_types
            
            return path
            
        except Exception as e:
            print(f"  Warning: Advanced silhouette extraction failed: {e}")
            return self._create_alpha_shape_projection(mesh.vertices[:, :2])
    
    def _create_alpha_shape_projection(self, vertices_2d):
        """
        Create more accurate projection using alpha shapes
        Better than convex hull for concave shapes
        """
        try:
            from scipy.spatial import Delaunay
            from scipy.spatial import ConvexHull
            
            if len(vertices_2d) < 4:
                # Too few points, use convex hull
                hull = ConvexHull(vertices_2d)
                hull_points = vertices_2d[hull.vertices]
                path = trimesh.path.Path2D(entities=[
                    trimesh.path.entities.Line(points=np.arange(len(hull_points)))
                ], vertices=hull_points)
                return path
            
            # Create Delaunay triangulation
            tri = Delaunay(vertices_2d)
            
            # Calculate alpha value adaptively based on local edge density
            # Use percentile-based approach for better robustness
            edge_lengths = []
            for simplex in tri.simplices:
                for i in range(3):
                    p1 = vertices_2d[simplex[i]]
                    p2 = vertices_2d[simplex[(i+1)%3]]
                    edge_lengths.append(np.linalg.norm(p2 - p1))
            
            # Use 85th percentile for alpha to capture more detail (was 75th)
            # More aggressive for complex boundaries
            alpha = np.percentile(edge_lengths, 85) * 1.5  # Increased multiplier from 1.3
            
            # Find boundary edges (alpha shape boundary)
            boundary_edges = []
            
            for simplex in tri.simplices:
                for i in range(3):
                    edge = [simplex[i], simplex[(i+1)%3]]
                    edge.sort()  # Normalize edge direction
                    
                    # Calculate circumradius of triangle
                    p1 = vertices_2d[simplex[0]]
                    p2 = vertices_2d[simplex[1]]
                    p3 = vertices_2d[simplex[2]]
                    
                    # Calculate circumradius using Heron's formula
                    a = np.linalg.norm(p2 - p3)
                    b = np.linalg.norm(p1 - p3)
                    c = np.linalg.norm(p1 - p2)
                    
                    s = (a + b + c) / 2
                    area = np.sqrt(max(0, s * (s-a) * (s-b) * (s-c)))
                    
                    if area > 1e-10:  # Avoid division by zero
                        circumradius = (a * b * c) / (4 * area)
                        
                        # Include edge if circumradius is small enough
                        if circumradius < alpha:
                            boundary_edges.append(edge)
            
            if not boundary_edges:
                # Fallback to convex hull
                hull = ConvexHull(vertices_2d)
                hull_points = vertices_2d[hull.vertices]
                path = trimesh.path.Path2D(entities=[
                    trimesh.path.entities.Line(points=np.arange(len(hull_points)))
                ], vertices=hull_points)
                return path
            
            # Remove duplicate edges and find actual boundary
            edge_count = {}
            for edge in boundary_edges:
                edge_tuple = tuple(edge)
                edge_count[edge_tuple] = edge_count.get(edge_tuple, 0) + 1
            
            # Boundary edges appear only once
            actual_boundary = [list(edge) for edge, count in edge_count.items() if count == 1]
            
            if not actual_boundary:
                # Fallback to convex hull
                hull = ConvexHull(vertices_2d)
                hull_points = vertices_2d[hull.vertices]
                path = trimesh.path.Path2D(entities=[
                    trimesh.path.entities.Line(points=np.arange(len(hull_points)))
                ], vertices=hull_points)
                return path
            
            # Create Path2D from boundary edges
            entities = []
            for edge in actual_boundary:
                entities.append(trimesh.path.entities.Line(points=edge))
            
            path = trimesh.path.Path2D(
                entities=entities,
                vertices=vertices_2d
            )
            
            return path
            
        except Exception as e:
            print(f"  Warning: Alpha shape failed, using convex hull: {e}")
            # Final fallback to convex hull
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(vertices_2d)
                hull_points = vertices_2d[hull.vertices]
                
                path = trimesh.path.Path2D(entities=[
                    trimesh.path.entities.Line(points=np.arange(len(hull_points)))
                ], vertices=hull_points)
                return path
            except:
                return None
    
    def _add_section_cuts(self, mesh, view, plane_position='center'):
        """
        Add section cuts to show internal features
        Creates cross-sections at the center or specified position
        
        Args:
            mesh: Input mesh (already rotated to view orientation)
            view: View name ('front', 'side', 'right', 'top', etc.)
            plane_position: 'center' or a specific coordinate value
            
        Returns:
            Path2D containing section cut edges
        """
        try:
            # Get mesh bounds
            bounds = mesh.bounds
            center = (bounds[0] + bounds[1]) / 2
            
            # Determine section plane based on view
            # The mesh is already rotated, so we cut perpendicular to the view direction
            section = None
            
            if view in ['front', 'back']:
                # For front/back views, cut along Y axis (depth)
                # This shows what's inside when looking from front
                plane_origin = [center[0], center[1], center[2]]
                plane_normal = [0, 1, 0]  # Cut perpendicular to Y
                
            elif view in ['side', 'right', 'left']:
                # For side views, cut along X axis (width)
                plane_origin = [center[0], center[1], center[2]]
                plane_normal = [1, 0, 0]  # Cut perpendicular to X
                
            elif view in ['top', 'bottom']:
                # For top/bottom views, cut along Z axis (height)
                plane_origin = [center[0], center[1], center[2]]
                plane_normal = [0, 0, 1]  # Cut perpendicular to Z
                
            else:
                # For other views, try a vertical cut
                plane_origin = [center[0], center[1], center[2]]
                plane_normal = [0, 1, 0]
            
            # Create the section
            section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
            
            if section is None:
                return None
            
            # Convert section to Path2D
            if hasattr(section, 'vertices') and len(section.vertices) > 0:
                # Project to 2D (take X,Y coordinates)
                vertices_2d = section.vertices[:, :2]
                
                # Create entities
                entities = []
                if hasattr(section, 'entities'):
                    entities = section.entities
                else:
                    # If no entities, create lines from vertices
                    # Assume vertices form a closed loop or multiple loops
                    if len(vertices_2d) >= 2:
                        # Simple line segments
                        for i in range(len(vertices_2d) - 1):
                            entities.append(trimesh.path.entities.Line(points=[i, i+1]))
                
                if not entities:
                    return None
                
                # Create Path2D
                path = trimesh.path.Path2D(
                    entities=entities,
                    vertices=vertices_2d
                )
                
                # Mark this as a section cut for rendering purposes
                if hasattr(path, 'metadata'):
                    path.metadata['is_section'] = True
                else:
                    path.metadata = {'is_section': True}
                
                return path
            
            return None
            
        except Exception as e:
            print(f"  Note: Section cuts not available: {e}")
            return None
    
    def _add_section_hatching(self, section_path, bounds, spacing=None):
        """
        Add hatching pattern to section cuts (standard CAD practice)
        
        Args:
            section_path: Path2D containing section cut edges
            bounds: Bounding box of the section [min, max]
            spacing: Spacing between hatch lines (auto-calculated if None)
            
        Returns:
            Path2D containing hatching lines
        """
        try:
            if section_path is None or not hasattr(section_path, 'vertices'):
                return None
            
            # Get bounds of section
            vertices = section_path.vertices
            if len(vertices) == 0:
                return None
            
            v_min = vertices.min(axis=0)
            v_max = vertices.max(axis=0)
            
            # Calculate spacing if not provided (based on size)
            if spacing is None:
                size = max(v_max[0] - v_min[0], v_max[1] - v_min[1])
                spacing = size * 0.05  # 5% of size
            
            # Create 45-degree hatch lines
            hatch_entities = []
            hatch_vertices = []
            
            # Determine hatch line range
            diagonal_size = np.linalg.norm(v_max - v_min)
            num_lines = int(diagonal_size / spacing) + 1
            
            # Generate hatch lines at 45 degrees
            for i in range(num_lines):
                # Start from bottom-left, go to top-right
                offset = v_min[0] + v_min[1] + i * spacing
                
                # Line equation: y = x - offset (45 degrees)
                # Find intersections with bounding box
                x_start = v_min[0]
                y_start = x_start + offset
                
                x_end = v_max[0]
                y_end = x_end + offset
                
                # Clip to bounds
                if y_start < v_min[1]:
                    y_start = v_min[1]
                    x_start = y_start - offset
                if y_start > v_max[1]:
                    y_start = v_max[1]
                    x_start = y_start - offset
                    
                if y_end < v_min[1]:
                    y_end = v_min[1]
                    x_end = y_end - offset
                if y_end > v_max[1]:
                    y_end = v_max[1]
                    x_end = y_end - offset
                
                # Check if line is within bounds
                if x_start >= v_min[0] and x_start <= v_max[0] and \
                   x_end >= v_min[0] and x_end <= v_max[0]:
                    # Add hatch line
                    idx_start = len(hatch_vertices)
                    hatch_vertices.append([x_start, y_start])
                    hatch_vertices.append([x_end, y_end])
                    hatch_entities.append(trimesh.path.entities.Line(points=[idx_start, idx_start + 1]))
            
            if not hatch_entities:
                return None
            
            # Create hatching Path2D
            hatch_path = trimesh.path.Path2D(
                entities=hatch_entities,
                vertices=np.array(hatch_vertices)
            )
            
            # Mark as hatching for rendering
            hatch_path.metadata = {'is_hatching': True}
            
            return hatch_path
            
        except Exception as e:
            print(f"  Note: Hatching generation failed: {e}")
            return None
    
    def _create_section_view(self, mesh, view):
        """
        Create a complete section view combining silhouette and section cut
        
        Args:
            mesh: Input mesh (already rotated to view orientation)
            view: View name
            
        Returns:
            Path2D containing complete section view
        """
        try:
            # Get the standard projection (silhouette)
            view_params = {
                'rotation': np.eye(4),
                'edge_angle': 5.0,
                'feature_scale': 2.5,
                'silhouette_threshold': 0.002,
                'min_edge_length': 0.00005
            }
            
            silhouette = self._extract_silhouette_edges(
                mesh,
                edge_angle=view_params['edge_angle'],
                feature_scale=view_params['feature_scale'],
                silhouette_threshold=view_params['silhouette_threshold'],
                min_edge_length=view_params.get('min_edge_length', 0.001)
            )
            
            # Get section cut
            section = self._add_section_cuts(mesh, view)
            
            # Combine silhouette and section
            if section and silhouette:
                combined = self._combine_projections(silhouette, section)
                
                # Add hatching to section areas
                if hasattr(section, 'vertices') and len(section.vertices) > 0:
                    bounds = [section.vertices.min(axis=0), section.vertices.max(axis=0)]
                    hatching = self._add_section_hatching(section, bounds)
                    
                    if hatching:
                        # Combine with hatching
                        combined = self._combine_projections(combined, hatching)
                
                return combined
            
            return silhouette or section
            
        except Exception as e:
            print(f"  Warning: Section view creation failed: {e}")
            return None

    
    def _combine_projections(self, silhouette, sections):
        """
        Combine silhouette edges with section cuts
        """
        try:
            # Combine entities from both projections
            all_entities = []
            all_vertices = []
            
            # Add silhouette entities
            if hasattr(silhouette, 'entities') and hasattr(silhouette, 'vertices'):
                all_vertices.extend(silhouette.vertices)
                all_entities.extend(silhouette.entities)
                vertex_offset = len(silhouette.vertices)
            else:
                vertex_offset = 0
            
            # Add section entities with offset
            if hasattr(sections, 'entities') and hasattr(sections, 'vertices'):
                all_vertices.extend(sections.vertices)
                
                for entity in sections.entities:
                    if hasattr(entity, 'points'):
                        adjusted_points = entity.points + vertex_offset
                        all_entities.append(trimesh.path.entities.Line(points=adjusted_points))
            
            if not all_entities:
                return silhouette
            
            # Create combined Path2D
            path = trimesh.path.Path2D(
                entities=all_entities,
                vertices=np.array(all_vertices)
            )
            
            return path
            
        except Exception as e:
            print(f"  Warning: Could not combine projections: {e}")
            return silhouette
    
    def _draw_dimensions(self, draw, projection, offset_x, offset_y, 
                        proj_center, scale_factor, font, line_width, view, axis_font, max_model_size):
        """
        Draw dimension lines and measurements on the projection
        Professional CAD-style with dashed bounding box and external dimensions
        
        Args:
            draw: ImageDraw object
            projection: Path2D projection object
            offset_x, offset_y: Center position for the view
            proj_center: Center of projection vertices
            scale_factor: Scaling factor applied to vertices
            font: Font for dimension text
            line_width: Line width for dimension lines
            view: View name (for determining which dimensions to show)
            axis_font: Font for axis labels
            max_model_size: Maximum model size for consistent box sizing
        """
        # Get projection bounds
        proj_verts = projection.vertices
        proj_min = proj_verts.min(axis=0)
        proj_max = proj_verts.max(axis=0)
        
        # Calculate actual dimensions in model units
        width = proj_max[0] - proj_min[0]
        height = proj_max[1] - proj_min[1]
        
        # Transform bounds to screen coordinates
        def to_screen(point):
            p = point.copy()
            p -= proj_center
            p = p * scale_factor
            screen_x = offset_x + p[0]
            screen_y = offset_y - p[1]  # Flip Y
            return (screen_x, screen_y)
        
        # Get corner points
        top_left = to_screen(np.array([proj_min[0], proj_max[1]]))
        top_right = to_screen(np.array([proj_max[0], proj_max[1]]))
        bottom_left = to_screen(np.array([proj_min[0], proj_min[1]]))
        bottom_right = to_screen(np.array([proj_max[0], proj_min[1]]))
        
        # Dimension settings - use consistent sizing across all views
        # Calculate dim_offset based on max_model_size instead of individual view dimensions
        dim_offset = scale_factor * max_model_size * 0.18
        arrow_size = line_width * 8.0
        dim_color = 'black'  # Reverted to black per user request
        box_color = (0, 0, 0)  # Black for inner bounding box
        
        # Calculate outer box dynamically based on dimension offset
        # The outer box must contain: dimensions + arrows + text boxes + axis labels
        # Add extra space for dimension text boxes and axis labels
        # Use consistent sizing across all views
        gap_x = max((dim_offset * 2.5) / scale_factor, max_model_size * 0.35)  # Use max_model_size for consistency
        gap_y = max((dim_offset * 2.5) / scale_factor, max_model_size * 0.35)
        
        # Outer box corners (expanded)
        outer_top_left = to_screen(np.array([proj_min[0] - gap_x, proj_max[1] + gap_y]))
        outer_top_right = to_screen(np.array([proj_max[0] + gap_x, proj_max[1] + gap_y]))
        outer_bottom_left = to_screen(np.array([proj_min[0] - gap_x, proj_min[1] - gap_y]))
        outer_bottom_right = to_screen(np.array([proj_max[0] + gap_x, proj_min[1] - gap_y]))
        
        # Draw outer box with axis grid (solid frame)
        outer_proj_min = np.array([proj_min[0] - gap_x, proj_min[1] - gap_y])
        outer_proj_max = np.array([proj_max[0] + gap_x, proj_max[1] + gap_y])
        self._draw_axis_grid(draw, outer_top_left, outer_bottom_right, 
                           outer_proj_min, outer_proj_max,
                           axis_font, line_width, '', '')
        
        # Draw inner dashed bounding box around the object
        dash_length = int(line_width * 4)
        gap_length = int(line_width * 3)
        self._draw_dashed_rectangle(draw, top_left, bottom_right, 
                                    box_color, int(line_width * 2), 
                                    dash_length, gap_length)
        
        # Horizontal dimension (width) - below the object
        h_dim_y = bottom_left[1] + dim_offset
        
        # Extension lines from bounding box corners
        ext_line_start = dim_offset * 0.3
        ext_line_end = dim_offset * 1.1
        
        draw.line([(bottom_left[0], bottom_left[1] + ext_line_start), 
                  (bottom_left[0], h_dim_y + arrow_size)], 
                 fill=dim_color, width=max(1, line_width // 2))
        draw.line([(bottom_right[0], bottom_right[1] + ext_line_start), 
                  (bottom_right[0], h_dim_y + arrow_size)], 
                 fill=dim_color, width=max(1, line_width // 2))
        
        # Dimension line with arrows at the ends
        arrow_offset = arrow_size * 1.2
        draw.line([(bottom_left[0] + arrow_offset, h_dim_y), (bottom_right[0] - arrow_offset, h_dim_y)], 
                 fill=dim_color, width=line_width)
        
        # Draw arrows pointing outward at the line ends
        self._draw_arrow(draw, (bottom_left[0] + arrow_offset, h_dim_y), 
                        (bottom_left[0], h_dim_y), arrow_size, dim_color, line_width)
        self._draw_arrow(draw, (bottom_right[0] - arrow_offset, h_dim_y), 
                        (bottom_right[0], h_dim_y), arrow_size, dim_color, line_width)
        
        # Dimension text in a box
        dim_text = f"{width:.2f}"
        text_pos = ((bottom_left[0] + bottom_right[0]) / 2, h_dim_y + arrow_size * 3)
        
        # Draw text background box
        bbox = draw.textbbox(text_pos, dim_text, font=font, anchor='mm')
        padding = line_width * 2
        draw.rectangle([(bbox[0] - padding, bbox[1] - padding), 
                       (bbox[2] + padding, bbox[3] + padding)], 
                      fill='white', outline=dim_color, width=max(1, line_width // 2))
        draw.text(text_pos, dim_text, fill=dim_color, font=font, anchor='mm')
        
        # Vertical dimension (height) - left of the object
        v_dim_x = top_left[0] - dim_offset
        
        # Extension lines from bounding box corners
        draw.line([(top_left[0] - ext_line_start, top_left[1]), 
                  (v_dim_x - arrow_size, top_left[1])], 
                 fill=dim_color, width=max(1, line_width // 2))
        draw.line([(bottom_left[0] - ext_line_start, bottom_left[1]), 
                  (v_dim_x - arrow_size, bottom_left[1])], 
                 fill=dim_color, width=max(1, line_width // 2))
        
        # Dimension line with arrows at the ends (closer to the object)
        draw.line([(v_dim_x, top_left[1] + arrow_offset), (v_dim_x, bottom_left[1] - arrow_offset)], 
                 fill=dim_color, width=line_width)
        
        # Draw arrows pointing outward at the line ends (closer to the object)
        self._draw_arrow(draw, (v_dim_x, top_left[1] + arrow_offset), 
                        (v_dim_x, top_left[1]), arrow_size, dim_color, line_width)
        self._draw_arrow(draw, (v_dim_x, bottom_left[1] - arrow_offset), 
                        (v_dim_x, bottom_left[1]), arrow_size, dim_color, line_width)
        
        # Dimension text in a box (positioned to the left of the dimension line)
        text_pos = (v_dim_x - arrow_size * 1.5, (top_left[1] + bottom_left[1]) / 2)
        dim_text_v = f"{height:.2f}"
        
        # Draw text background box (right-aligned to the dimension line)
        bbox = draw.textbbox(text_pos, dim_text_v, font=font, anchor='rm')
        padding = line_width * 2
        draw.rectangle([(bbox[0] - padding, bbox[1] - padding), 
                       (bbox[2] + padding, bbox[3] + padding)], 
                      fill='white', outline=dim_color, width=max(1, line_width // 2))
        draw.text(text_pos, dim_text_v, fill=dim_color, font=font, anchor='rm')
        
        # Add axis labels based on view
        axis_color = 'black'
        
        # Determine axis labels based on view
        if view == 'front':
            x_label = 'X-axis (units)'
            y_label = 'Z-axis (units)'
        elif view == 'top':
            x_label = 'X-axis (units)'
            y_label = 'Y-axis (units)'
        elif view == 'side':
            x_label = 'Y-axis (units)'
            y_label = 'Z-axis (units)'
        else:  # isometric or other
            x_label = 'X-axis (units)'
            y_label = 'Y-axis (units)'
        
        # X-axis label (below outer box, close to frame)
        # Position below the outer box bottom edge, with space for tick labels
        x_label_pos = ((outer_bottom_left[0] + outer_bottom_right[0]) / 2, outer_bottom_right[1] + line_width * 35)
        draw.text(x_label_pos, x_label, fill=axis_color, font=axis_font, anchor='mt')
        
        # Y-axis label (right of outer box, close to frame)
        # Position just to the right of the outer box right edge
        y_label_pos = (outer_top_right[0] + line_width * 15, (outer_top_right[1] + outer_bottom_right[1]) / 2)
        draw.text(y_label_pos, y_label, fill=axis_color, font=axis_font, anchor='lm')
    
    def _draw_axis_grid(self, draw, top_left, bottom_right, proj_min, proj_max,
                       font, line_width, x_label, y_label):
        """
        Draw axis grid with tick marks and numerical labels
        Similar to matplotlib style
        """
        axis_color = 'black'
        grid_color = (0, 0, 0)  # Black for outer box
        tick_length = line_width * 4
        
        # Calculate number of ticks (approximately 3-5 ticks per axis)
        num_ticks = 4
        
        # X-axis (bottom)
        x_range = proj_max[0] - proj_min[0]
        x_tick_step = x_range / (num_ticks - 1)
        
        for i in range(num_ticks):
            # Calculate position
            x_val = proj_min[0] + i * x_tick_step
            x_pos = top_left[0] + (bottom_right[0] - top_left[0]) * i / (num_ticks - 1)
            
            # Draw tick mark
            draw.line([(x_pos, bottom_right[1]), (x_pos, bottom_right[1] + tick_length)],
                     fill=axis_color, width=max(1, line_width // 2))
            
            # Draw grid line (vertical) - DISABLED
            # if i > 0 and i < num_ticks - 1:  # Skip first and last
            #     self._draw_dashed_line(draw, (x_pos, top_left[1]), (x_pos, bottom_right[1]),
            #                           grid_color, max(1, line_width // 3), 
            #                           line_width * 2, line_width * 2)
            
            # Draw tick label with more vertical spacing to avoid overlap with dimensions
            label_y = bottom_right[1] + tick_length + line_width * 8  # Increased from 4 to 8
            draw.text((x_pos, label_y), f"{x_val:.1f}", 
                     fill=axis_color, font=font, anchor='mt')
        
        # Y-axis (left)
        y_range = proj_max[1] - proj_min[1]
        y_tick_step = y_range / (num_ticks - 1)
        
        for i in range(num_ticks):
            # Calculate position (inverted because screen Y is flipped)
            y_val = proj_min[1] + i * y_tick_step
            y_pos = bottom_right[1] - (bottom_right[1] - top_left[1]) * i / (num_ticks - 1)
            
            # Draw tick mark
            draw.line([(top_left[0] - tick_length, y_pos), (top_left[0], y_pos)],
                     fill=axis_color, width=max(1, line_width // 2))
            
            # Draw grid line (horizontal) - DISABLED
            # if i > 0 and i < num_ticks - 1:  # Skip first and last
            #     self._draw_dashed_line(draw, (top_left[0], y_pos), (bottom_right[0], y_pos),
            #                           grid_color, max(1, line_width // 3),
            #                           line_width * 2, line_width * 2)
            
            # Draw tick label with more horizontal spacing to avoid overlap with dimensions
            label_x = top_left[0] - tick_length - line_width * 4  # Increased from 2 to 4
            draw.text((label_x, y_pos), f"{y_val:.1f}", 
                     fill=axis_color, font=font, anchor='rm')
        
        # Draw axis lines (frame)
        # Left axis
        draw.line([(top_left[0], top_left[1]), (top_left[0], bottom_right[1])],
                 fill=axis_color, width=line_width)
        # Bottom axis
        draw.line([(top_left[0], bottom_right[1]), (bottom_right[0], bottom_right[1])],
                 fill=axis_color, width=line_width)
        # Right axis (lighter)
        draw.line([(bottom_right[0], top_left[1]), (bottom_right[0], bottom_right[1])],
                 fill=grid_color, width=max(1, line_width // 2))
        # Top axis (lighter)
        draw.line([(top_left[0], top_left[1]), (bottom_right[0], top_left[1])],
                 fill=grid_color, width=max(1, line_width // 2))
    
    def _draw_dashed_rectangle(self, draw, top_left, bottom_right, color, width, dash_len, gap_len):
        """
        Draw a dashed rectangle (bounding box)
        """
        corners = [
            top_left,
            (bottom_right[0], top_left[1]),
            bottom_right,
            (top_left[0], bottom_right[1])
        ]
        
        # Draw each side as dashed line
        for i in range(4):
            start = corners[i]
            end = corners[(i + 1) % 4]
            self._draw_dashed_line(draw, start, end, color, width, dash_len, gap_len)
    
    def _draw_dashed_line(self, draw, start, end, color, width, dash_len=None, gap_len=None, pattern=None):
        """
        Draw a dashed line between two points with optional pattern
        pattern: list of [dash, gap, dash, gap, ...] lengths
        """
        x1, y1 = start
        x2, y2 = end
        
        # Calculate line length and direction
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        
        if length < 0.001:
            return
        
        # Normalize direction
        dx /= length
        dy /= length
        
        # Use pattern or simple dash/gap
        if pattern is None:
            if dash_len is None: dash_len = width * 4
            if gap_len is None: gap_len = width * 2
            pattern = [dash_len, gap_len]
            
        # Draw dashes based on pattern
        current_pos = 0
        pat_idx = 0
        
        while current_pos < length:
            # Get current segment length from pattern
            seg_len = pattern[pat_idx % len(pattern)]
            
            # Draw if it's a dash (even index)
            if pat_idx % 2 == 0:
                # Start of dash
                dash_start_x = x1 + dx * current_pos
                dash_start_y = y1 + dy * current_pos
                
                # End of dash
                dash_end_pos = min(current_pos + seg_len, length)
                dash_end_x = x1 + dx * dash_end_pos
                dash_end_y = y1 + dy * dash_end_pos
                
                # Draw dash
                draw.line([(dash_start_x, dash_start_y), (dash_end_x, dash_end_y)], 
                         fill=color, width=width)
            
            # Move to next segment
            current_pos += seg_len
            pat_idx += 1
    
    def _draw_arrow(self, draw, start, end, size, color, line_width):
        """
        Draw an arrow from start to end point
        """
        # Calculate arrow direction
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = np.sqrt(dx**2 + dy**2)
        
        if length < 0.001:
            return
        
        # Normalize
        dx /= length
        dy /= length
        
        # Perpendicular vector
        px = -dy
        py = dx
        
        # Arrow head points
        arrow_width = size * 0.5
        p1 = (end[0] - dx * size + px * arrow_width, 
              end[1] - dy * size + py * arrow_width)
        p2 = (end[0] - dx * size - px * arrow_width, 
              end[1] - dy * size - py * arrow_width)
        
        # Draw filled triangle
        draw.polygon([end, p1, p2], fill=color)
    
    def export_to_dxf(self, output_path, views=['front', 'top', 'side']):
        """
        Export 2D projections to DXF format with professional layout and layers
        
        Args:
            output_path: Output file path
            views: List of views to include
        """
        try:
            import ezdxf
            from ezdxf import units
        except ImportError:
            print("✗ ezdxf not installed. Install with: pip install ezdxf")
            return
        
        # Create a new DXF document (R12 version for maximum compatibility)
        doc = ezdxf.new('R12')
        doc.units = units.MM
        msp = doc.modelspace()
        
        # Define layers that will be referenced later (ShareCAD requires explicit entries)
        required_layers = ['VISIBLE', 'HIDDEN', 'CENTER', 'DIMENSIONS', 'TEXT']
        for layer_name in required_layers:
            if layer_name not in doc.layers:
                doc.layers.new(name=layer_name)
            
        # Calculate grid layout
        num_views = len(views)
        if num_views <= 2:
            grid_cols = 1
            grid_rows = num_views
        elif num_views <= 4:
            grid_cols = 2
            grid_rows = (num_views + 1) // 2
        elif num_views <= 6:
            grid_cols = 3
            grid_rows = (num_views + 2) // 3
        else:
            grid_cols = 3
            grid_rows = (num_views + 2) // 3
            
        # Calculate consistent spacing based on maximum extents across all views
        max_model_size = 0
        view_projections = {}
        
        for view in views:
            projection = self.get_projection(view)
            if projection and hasattr(projection, 'vertices') and len(projection.vertices) > 0:
                proj_verts = projection.vertices
                proj_min = proj_verts.min(axis=0)
                proj_max = proj_verts.max(axis=0)
                width_dim = proj_max[0] - proj_min[0]
                height_dim = proj_max[1] - proj_min[1]
                max_extent = max(width_dim, height_dim)
                max_model_size = max(max_model_size, max_extent)
                view_projections[view] = projection
        
        # If no valid projections found, use fallback
        if max_model_size == 0:
            max_model_size = self.mesh.bounding_box.extents.max()
            
        # Use consistent cell size based on maximum model size
        cell_width = max_model_size * 2.5
        cell_height = max_model_size * 2.5
        
        # Margins
        start_x = 0
        start_y = 0
        
        for idx, view in enumerate(views):
            print(f"  Generating {view} view...")
            # Use pre-computed projection if available, otherwise get it
            projection = view_projections.get(view, self.get_projection(view))
            
            if projection is None:
                continue
                
            # Calculate grid position
            col = idx % grid_cols
            row = idx // grid_cols
            
            # Center position for this view in the grid
            # DXF coordinates: Y is up, so we stack rows downwards (negative Y)
            offset_x = start_x + col * cell_width + cell_width / 2
            offset_y = start_y - (row * cell_height + cell_height / 2)
            
            # Get projection bounds
            if hasattr(projection, 'vertices') and len(projection.vertices) > 0:
                proj_verts = projection.vertices
                proj_min = proj_verts.min(axis=0)
                proj_max = proj_verts.max(axis=0)
                proj_center = (proj_min + proj_max) / 2
                width_dim = proj_max[0] - proj_min[0]
                height_dim = proj_max[1] - proj_min[1]
                
                # Draw entities
                if hasattr(projection, 'entities'):
                    # Get edge types if available
                    edge_types = {}
                    if hasattr(projection, 'metadata') and 'edge_types' in projection.metadata:
                        edge_types = projection.metadata['edge_types']
                    
                    for entity in projection.entities:
                        # Get points for this entity
                        points = projection.vertices[entity.points]
                        
                        # Center and position
                        points_centered = points - proj_center
                        points_final = points_centered.copy()
                        points_final[:, 0] += offset_x
                        points_final[:, 1] += offset_y
                        
                        # Determine layer based on edge type
                        layer = 'VISIBLE'
                        # Check if this edge matches a known type
                        # This is approximate as we're matching float coordinates or indices
                        # For now, we'll default to VISIBLE unless we can map it back
                        # A better way is to store the layer in the entity itself during creation
                        
                        # Draw lines without layer specification for maximum compatibility
                        for i in range(len(points_final) - 1):
                            msp.add_line(
                                (float(points_final[i][0]), float(points_final[i][1])),
                                (float(points_final[i+1][0]), float(points_final[i+1][1]))
                            )
                        
                        # Close loop if needed
                        if hasattr(entity, 'closed') and entity.closed:
                            msp.add_line(
                                (float(points_final[-1][0]), float(points_final[-1][1])),
                                (float(points_final[0][0]), float(points_final[0][1]))
                            )
                
                                
                # Add Dimensions
                dim_offset = max_model_size * 0.1
                
                # Simple dimensions without advanced styling for better compatibility
                # Width Dimension (Horizontal) - simple lines and text
                msp.add_line(
                    (offset_x - width_dim/2, offset_y - height_dim/2 - dim_offset),
                    (offset_x + width_dim/2, offset_y - height_dim/2 - dim_offset),
                    dxfattribs={'layer': 'DIMENSIONS'}
                )
                # Extension lines
                msp.add_line(
                    (offset_x - width_dim/2, offset_y - height_dim/2),
                    (offset_x - width_dim/2, offset_y - height_dim/2 - dim_offset),
                    dxfattribs={'layer': 'DIMENSIONS'}
                )
                msp.add_line(
                    (offset_x + width_dim/2, offset_y - height_dim/2),
                    (offset_x + width_dim/2, offset_y - height_dim/2 - dim_offset),
                    dxfattribs={'layer': 'DIMENSIONS'}
                )
                # Dimension text
                msp.add_text(
                    f"{width_dim:.2f}",
                    dxfattribs={'layer': 'TEXT', 'height': 2.5}
                ).set_placement(
                    (offset_x, offset_y - height_dim/2 - dim_offset - 2),
                    align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER
                )
                
                # Height Dimension (Vertical) - simple lines and text
                msp.add_line(
                    (offset_x - width_dim/2 - dim_offset, offset_y - height_dim/2),
                    (offset_x - width_dim/2 - dim_offset, offset_y + height_dim/2),
                    dxfattribs={'layer': 'DIMENSIONS'}
                )
                # Extension lines
                msp.add_line(
                    (offset_x - width_dim/2, offset_y - height_dim/2),
                    (offset_x - width_dim/2 - dim_offset, offset_y - height_dim/2),
                    dxfattribs={'layer': 'DIMENSIONS'}
                )
                msp.add_line(
                    (offset_x - width_dim/2, offset_y + height_dim/2),
                    (offset_x - width_dim/2 - dim_offset, offset_y + height_dim/2),
                    dxfattribs={'layer': 'DIMENSIONS'}
                )
                # Dimension text
                msp.add_text(
                    f"{height_dim:.2f}",
                    dxfattribs={'layer': 'TEXT', 'height': 2.5}
                ).set_placement(
                    (offset_x - width_dim/2 - dim_offset - 2, offset_y),
                    align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER
                )
                
                # Add Centerlines
                center_len = max_model_size * 0.02
                # Vertical
                msp.add_line(
                    (offset_x, offset_y - height_dim/2 - center_len),
                    (offset_x, offset_y + height_dim/2 + center_len),
                    dxfattribs={'layer': 'CENTER'}
                )
                # Horizontal
                msp.add_line(
                    (offset_x - width_dim/2 - center_len, offset_y),
                    (offset_x + width_dim/2 + center_len, offset_y),
                    dxfattribs={'layer': 'CENTER'}
                )
    
        doc.saveas(output_path)
        print(f"[+] Saved DXF: {output_path}")
        print(f"  Layers: VISIBLE, HIDDEN, CENTER, DIMENSIONS, TEXT")
        print(f"  Layout: {grid_cols}x{grid_rows} Grid")
    
    def export_to_svg(self, output_path, views=['front', 'top', 'side']):
        """
        Export 2D projections to SVG format
        
        Args:
            output_path: Output file path
            views: List of views to include
        """
        import svgwrite
        
        # Calculate layout dimensions with consistent sizing
        # Get maximum extents across all views for consistent spacing
        max_model_size = 0
        view_projections = {}
        
        for view in views:
            projection = self.get_projection(view)
            if projection and hasattr(projection, 'vertices') and len(projection.vertices) > 0:
                proj_verts = projection.vertices
                proj_min = proj_verts.min(axis=0)
                proj_max = proj_verts.max(axis=0)
                width_dim = proj_max[0] - proj_min[0]
                height_dim = proj_max[1] - proj_min[1]
                max_extent = max(width_dim, height_dim)
                max_model_size = max(max_model_size, max_extent)
                view_projections[view] = projection
        
        # If no valid projections found, use fallback
        if max_model_size == 0:
            max_model_size = self.mesh.bounding_box.extents.max()
            
        spacing = max_model_size * 1.5
        width = spacing * len(views)
        height = spacing * 1.5
        
        dwg = svgwrite.Drawing(output_path, size=(f'{width}mm', f'{height}mm'),
                               viewBox=f'0 0 {width} {height}')
        
        offset_x = spacing * 0.25
        
        for view in views:
            print(f"  Generating {view} view...")
            # Use pre-computed projection if available, otherwise get it
            projection = view_projections.get(view, self.get_projection(view))
            
            if projection is None:
                print(f"    Warning: Could not generate {view} view")
                continue
            
            # Create a group for this view
            view_group = dwg.g(id=f'{view}_view')
            
            # Add projection paths to SVG
            if hasattr(projection, 'entities') and hasattr(projection, 'vertices'):
                for entity in projection.entities:
                    # Get the actual point coordinates from vertices
                    if hasattr(entity, 'points'):
                        # entity.points contains indices into projection.vertices
                        point_indices = entity.points
                        if len(point_indices) > 0:
                            # Get actual coordinates
                            points = projection.vertices[point_indices].copy()
                            # Offset for layout
                            points[:, 0] += offset_x
                            points[:, 1] += spacing * 0.5
                            # Flip Y coordinate for SVG
                            points[:, 1] = height - points[:, 1]
                            
                            path_data = f"M {points[0][0]},{points[0][1]}"
                            for point in points[1:]:
                                path_data += f" L {point[0]},{point[1]}"
                            if hasattr(entity, 'closed') and entity.closed:
                                path_data += " Z"
                            
                            view_group.add(dwg.path(d=path_data,
                                                   stroke='black',
                                                   fill='none',
                                                   stroke_width=0.5))
            
            # Add view label
            view_group.add(dwg.text(view.upper(),
                                   insert=(offset_x, height - spacing * 0.1),
                                   font_size=spacing * 0.08,
                                   font_family='Arial'))
            
            dwg.add(view_group)
            offset_x += spacing
        
        dwg.save()
        print(f"[+] Saved SVG: {output_path}")
        
    def export_to_dxf_basic(self, output_path, views=['front', 'top', 'side']):
        """
        Export ultra-simple DXF for maximum compatibility (ShareCAD).
        - DXF R2000 format (better entity support than R12)
        - POLYLINE entities on layer 0
        - NO text, NO custom layers, NO attributes
        - Standard DXF structure for CAD viewers
        """
        try:
            import ezdxf
        except ImportError:
            print("[!] ezdxf not installed. Install with: pip install ezdxf")
            return

        # Create R2000 DXF - better support for polylines and modern viewers
        doc = ezdxf.new('R2000')
        msp = doc.modelspace()

        # Grid layout
        num_views = len(views)
        if num_views <= 2:
            grid_cols = 2
            grid_rows = 1
        elif num_views <= 4:
            grid_cols = 2
            grid_rows = 2
        else:
            grid_cols = 3
            grid_rows = (num_views + 2) // 3

        # Find max extent for uniform scaling
        pre_proj = {}
        max_extent = 0.0
        for v in views:
            proj = self.get_projection(v)
            if proj is None or not hasattr(proj, 'vertices') or len(proj.vertices) == 0:
                continue
            pre_proj[v] = proj
            verts = proj.vertices
            pmin = verts.min(axis=0)
            pmax = verts.max(axis=0)
            ext = max(float(pmax[0] - pmin[0]), float(pmax[1] - pmin[1]))
            if ext > max_extent:
                max_extent = ext

        if max_extent <= 0:
            max_extent = float(self.mesh.bounding_box.extents.max())

        # Cell size and scale
        cell_size = max_extent * 2.5
        scale = (cell_size * 0.6) / max_extent if max_extent > 0 else 1.0

        # Track actual geometry extents for DXF header
        global_min_x = float('inf')
        global_min_y = float('inf')
        global_max_x = float('-inf')
        global_max_y = float('-inf')
        has_geometry = False

        # Draw geometry
        for idx, v in enumerate(views):
            print(f"  Generating {v} view (basic DXF)...")
            proj = pre_proj.get(v, self.get_projection(v))
            if proj is None or not hasattr(proj, 'vertices') or len(proj.vertices) == 0:
                continue

            col = idx % grid_cols
            row = idx // grid_cols
            cx = col * cell_size + cell_size / 2.0
            cy = -(row * cell_size + cell_size / 2.0)

            verts = proj.vertices
            pmin = verts.min(axis=0)
            pmax = verts.max(axis=0)
            pc = (pmin + pmax) / 2.0

            if hasattr(proj, 'entities'):
                for ent in proj.entities:
                    if hasattr(ent, 'points') and len(ent.points) > 0:
                        pts = verts[ent.points].copy()
                        pts = (pts - pc) * scale
                        pts[:, 0] += cx
                        pts[:, 1] += cy
                        
                        # Use LWPOLYLINE (lightweight polyline) - modern DXF standard
                        # Convert to list of tuples for ezdxf
                        point_list = [(float(pts[i][0]), float(pts[i][1])) for i in range(len(pts))]
                        
                        # Filter out consecutive duplicate points
                        filtered_points = []
                        for i, pt in enumerate(point_list):
                            if i == 0 or (pt[0] != filtered_points[-1][0] or pt[1] != filtered_points[-1][1]):
                                filtered_points.append(pt)
                        
                        # Only add if we have at least 2 points
                        if len(filtered_points) >= 2:
                            # Add as LWPOLYLINE entity (R2000+ standard)
                            is_closed = hasattr(ent, 'closed') and ent.closed
                            try:
                                msp.add_lwpolyline(filtered_points, close=is_closed)
                                has_geometry = True
                            except:
                                # Fallback to regular polyline if LWPOLYLINE fails
                                msp.add_polyline2d(filtered_points, close=is_closed)
                                has_geometry = True
                            
                            # Update global extents
                            for pt in filtered_points:
                                global_min_x = min(global_min_x, pt[0])
                                global_min_y = min(global_min_y, pt[1])
                                global_max_x = max(global_max_x, pt[0])
                                global_max_y = max(global_max_y, pt[1])

        # Set DXF header extents so ShareCAD knows where to look
        if has_geometry and all(v != float('inf') and v != float('-inf') 
                                for v in [global_min_x, global_min_y, global_max_x, global_max_y]):
            # Add 10% padding for better visibility
            padding_x = (global_max_x - global_min_x) * 0.1
            padding_y = (global_max_y - global_min_y) * 0.1
            
            ext_min_x = global_min_x - padding_x
            ext_min_y = global_min_y - padding_y
            ext_max_x = global_max_x + padding_x
            ext_max_y = global_max_y + padding_y
            
            # Set drawing extents (tells CAD viewer where geometry is)
            doc.header['$EXTMIN'] = (ext_min_x, ext_min_y, 0.0)
            doc.header['$EXTMAX'] = (ext_max_x, ext_max_y, 0.0)
            
            # Set drawing limits (required by some viewers)
            doc.header['$LIMMIN'] = (ext_min_x, ext_min_y)
            doc.header['$LIMMAX'] = (ext_max_x, ext_max_y)
            
            # Set insertion base point to center
            doc.header['$INSBASE'] = ((ext_min_x + ext_max_x) / 2, (ext_min_y + ext_max_y) / 2, 0.0)
            
            print(f"  Drawing extents: ({ext_min_x:.2f}, {ext_min_y:.2f}) to ({ext_max_x:.2f}, {ext_max_y:.2f})")
        else:
            print(f"  Warning: No geometry found or invalid extents")

        # Audit the DXF document for errors before saving
        try:
            auditor = doc.audit()
            if auditor.has_errors:
                print(f"  Warning: DXF has {len(auditor.errors)} errors, attempting to fix...")
                auditor.print_error_report()
        except:
            pass  # Audit not critical, continue with save

        # Save with explicit file writing for better compatibility
        try:
            import os
            import time
            
            # Write DXF file with explicit encoding
            with open(output_path, 'wt', encoding='utf-8') as f:
                doc.write(f)
            
            # Verify file was written successfully
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                if file_size > 0:
                    # Small delay to ensure file system has finished writing
                    time.sleep(0.15)
                    print(f"[+] Saved BASIC DXF (R2000): {output_path}")
                    print(f"  Format: R2000 with POLYLINE entities")
                    print(f"  File size: {file_size} bytes")
                else:
                    print(f"[!] Warning: DXF file created but appears empty")
            else:
                print(f"[!] Error: Failed to create DXF file")
        except Exception as e:
            print(f"[!] Error saving DXF: {e}")
            # Fallback to saveas if write fails
            try:
                doc.saveas(output_path)
                print(f"[+] Saved BASIC DXF (R2000) via fallback: {output_path}")
            except Exception as e2:
                print(f"[!] Fallback save also failed: {e2}")
    
    def export_to_png(self, output_path, views=['front', 'top', 'side'], 
                      resolution=(5760, 3240), line_width=3, show_dimensions=True):
        """
        Export 2D projections to PNG format with high quality
        
        Args:
            output_path: Output file path
            views: List of views to include
            resolution: Output image resolution (width, height) - default 6K
            line_width: Line thickness in pixels (default 3)
            show_dimensions: Whether to show dimension annotations (default True)
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            print("[!] Pillow not installed. Install with: pip install Pillow")
            return
        
        # Create high-resolution image with anti-aliasing
        # Use 2x supersampling for better quality
        supersample = 2
        work_res = (resolution[0] * supersample, resolution[1] * supersample)
        img = Image.new('RGB', work_res, 'white')
        draw = ImageDraw.Draw(img)
        
        # Draw main title at the top
        title_size = int(work_res[1] * 0.035)  # 3.5% of image height
        try:
            title_font = ImageFont.truetype("arial.ttf", title_size)
        except:
            try:
                title_font = ImageFont.truetype("Arial.ttf", title_size)
            except:
                title_font = ImageFont.load_default()
        
        title_text = f"High-Accuracy Multi-View Technical Drawing - {self.model_path.name}"
        title_pos = (work_res[0] // 2, int(work_res[1] * 0.03))
        draw.text(title_pos, title_text, fill='black', font=title_font, anchor='mm')
        
        # Calculate grid layout for views
        # Determine grid dimensions based on number of views
        num_views = len(views)
        if num_views <= 2:
            grid_cols = 2
            grid_rows = 1
        elif num_views <= 4:
            grid_cols = 2
            grid_rows = 2
        elif num_views <= 6:
            grid_cols = 3
            grid_rows = 2
        else:
            grid_cols = 3
            grid_rows = (num_views + 2) // 3  # Ceiling division
        
        # Calculate layout with grid-based margins
        top_margin = int(work_res[1] * 0.08)  # 8% for title
        bottom_margin = int(work_res[1] * 0.05)  # 5% bottom margin
        left_margin = int(work_res[0] * 0.03)  # 3% left margin
        right_margin = int(work_res[0] * 0.03)  # 3% right margin
        
        # Available space for grid
        grid_width = work_res[0] - left_margin - right_margin
        grid_height = work_res[1] - top_margin - bottom_margin
        
        # Cell dimensions
        cell_width = grid_width // grid_cols
        cell_height = grid_height // grid_rows
        
        # Usable space within each cell (leave room for dimensions and labels)
        margin_factor = 0.35  # Increased from 0.30 to ensure labels don't overlap
        usable_width = cell_width * (1 - 2 * margin_factor)
        usable_height = cell_height * (1 - 2 * margin_factor)
        
        # Calculate scale to fit model in view - NORMALIZE ACROSS ALL VIEWS
        # Get the maximum extents across all views to ensure consistent sizing
        max_model_size = 0
        view_projections = {}
        
        for view in views:
            projection = self.get_projection(view)
            if projection and hasattr(projection, 'vertices') and len(projection.vertices) > 0:
                proj_verts = projection.vertices
                proj_min = proj_verts.min(axis=0)
                proj_max = proj_verts.max(axis=0)
                width_dim = proj_max[0] - proj_min[0]
                height_dim = proj_max[1] - proj_min[1]
                max_extent = max(width_dim, height_dim)
                max_model_size = max(max_model_size, max_extent)
                view_projections[view] = projection
        
        # If no valid projections found, use fallback
        if max_model_size == 0:
            max_model_size = self.mesh.bounding_box.extents.max()
        
        # Use the same scale factor for all views
        scale_factor = min(usable_width, usable_height) / max_model_size
        
        # Adjust line width for supersampling
        adjusted_line_width = line_width * supersample
        
        for idx, view in enumerate(views):
            print(f"  Generating {view} view...")
            # Use pre-computed projection if available, otherwise get it
            projection = view_projections.get(view, self.get_projection(view))
            
            if projection is None:
                print(f"    Warning: Could not generate {view} view")
                continue
            
            # Calculate grid position
            col = idx % grid_cols
            row = idx // grid_cols
            
            # Center position for this view in the grid
            offset_x = left_margin + col * cell_width + cell_width // 2
            offset_y = top_margin + row * cell_height + cell_height // 2
            
            # Draw view heading and dimensions summary
            heading_size = int(work_res[1] * 0.025)  # 2.5% of image height
            try:
                heading_font = ImageFont.truetype("arialbd.ttf", heading_size)  # Bold
            except:
                try:
                    heading_font = ImageFont.truetype("Arial.ttf", heading_size)
                except:
                    heading_font = ImageFont.load_default()
            
            # Get projection bounds for dimension summary
            if hasattr(projection, 'vertices') and len(projection.vertices) > 0:
                proj_verts = projection.vertices
                proj_min = proj_verts.min(axis=0)
                proj_max = proj_verts.max(axis=0)
                width_dim = proj_max[0] - proj_min[0]
                height_dim = proj_max[1] - proj_min[1]
                
                # Calculate projection center for alignment
                proj_center = (proj_min + proj_max) / 2
                
                # Calculate cell boundaries
                cell_top = top_margin + row * cell_height
                cell_left = left_margin + col * cell_width
                
                # View heading - positioned at top of cell with margin
                view_title = f"{view.upper()} VIEW"
                heading_y = cell_top + int(cell_height * 0.03)  # Moved up from 0.08 to 0.03
                draw.text((offset_x, heading_y), view_title, 
                         fill='black', font=heading_font, anchor='mm')
                
                # Dimension summary below heading
                dim_summary = f"{width_dim:.2f} × {height_dim:.2f}"
                summary_y = heading_y + heading_size + int(work_res[1] * 0.01)  # Below heading
                draw.text((offset_x, summary_y), dim_summary, 
                         fill='black', font=heading_font, anchor='mm')
                
                # Draw centerlines (axis lines) for symmetric objects
                # Draw if the view looks symmetric (simple heuristic) or always for standard views
                center_x = (proj_min[0] + proj_max[0]) / 2
                center_y = (proj_min[1] + proj_max[1]) / 2
                
                # Transform center to image coordinates
                img_center_x = offset_x + (center_x - proj_center[0]) * scale_factor
                img_center_y = offset_y - (center_y - proj_center[1]) * scale_factor
                
                # Draw vertical centerline (dash-dot)
                dash_len = int(work_res[0] * 0.01)
                gap_len = int(work_res[0] * 0.005)
                
                # Vertical line extending beyond bounds
                v_start_y = offset_y - (proj_max[1] - proj_center[1]) * scale_factor - dash_len
                v_end_y = offset_y - (proj_min[1] - proj_center[1]) * scale_factor + dash_len
                
                self._draw_dashed_line(draw, (img_center_x, v_start_y), (img_center_x, v_end_y),
                                     color='red', width=int(adjusted_line_width * 0.5), 
                                     pattern=[dash_len * 2, gap_len, dash_len // 2, gap_len])
                
                # Horizontal centerline
                h_start_x = offset_x + (proj_min[0] - proj_center[0]) * scale_factor - dash_len
                h_end_x = offset_x + (proj_max[0] - proj_center[0]) * scale_factor + dash_len
                
                self._draw_dashed_line(draw, (h_start_x, img_center_y), (h_end_x, img_center_y),
                                     color='red', width=int(adjusted_line_width * 0.5),
                                     pattern=[dash_len * 2, gap_len, dash_len // 2, gap_len])
            
            # Draw projection with anti-aliased lines
            if hasattr(projection, 'entities') and hasattr(projection, 'vertices'):
                # Get bounds of this projection for centering
                proj_verts = projection.vertices
                if len(proj_verts) > 0:
                    proj_min = proj_verts.min(axis=0)
                    proj_max = proj_verts.max(axis=0)
                    proj_center = (proj_min + proj_max) / 2
                    
                    for entity in projection.entities:
                        if hasattr(entity, 'points'):
                            # entity.points contains indices into projection.vertices
                            point_indices = entity.points
                            if len(point_indices) > 0:
                                # Get actual coordinates
                                points = projection.vertices[point_indices].copy()
                                
                                # Center the projection
                                points -= proj_center
                                
                                # Scale and position
                                points = points * scale_factor
                                points[:, 0] += offset_x
                                points[:, 1] = offset_y - points[:, 1]  # Flip Y
                                
                                point_list = [(float(p[0]), float(p[1])) for p in points]
                                if len(point_list) > 1:
                                    # Draw with thicker lines for visibility
                                    draw.line(point_list, fill='black', 
                                            width=adjusted_line_width, joint='curve')
                                    if hasattr(entity, 'closed') and entity.closed:
                                        draw.line([point_list[-1], point_list[0]], 
                                                fill='black', width=adjusted_line_width,
                                                joint='curve')
            
            # Dimension font (smaller)
            dim_size = int(work_res[1] * 0.02)  # 2% of image height
            try:
                dim_font = ImageFont.truetype("arial.ttf", dim_size)
            except:
                try:
                    dim_font = ImageFont.truetype("Arial.ttf", dim_size)
                except:
                    dim_font = ImageFont.load_default()
            
            # Axis label font
            axis_size = int(work_res[1] * 0.018)  # 1.8% of image height
            try:
                axis_font = ImageFont.truetype("arial.ttf", axis_size)
            except:
                try:
                    axis_font = ImageFont.truetype("Arial.ttf", axis_size)
                except:
                    axis_font = ImageFont.load_default()
            
            # Add dimensions if requested
            if show_dimensions and hasattr(projection, 'vertices') and len(projection.vertices) > 0:
                self._draw_dimensions(draw, projection, offset_x, offset_y, 
                                    proj_center, scale_factor, dim_font, 
                                    adjusted_line_width, view, axis_font, max_model_size)
        
        # Downsample to target resolution with high-quality anti-aliasing
        img = img.resize(resolution, Image.LANCZOS)
        
        # Save with high quality
        img.save(output_path, quality=95, optimize=True)
        print(f"[+] Saved PNG: {output_path}")
        print(f"  Resolution: {resolution[0]}x{resolution[1]}")
        print(f"  Line width: {line_width}px")
        print(f"  Anti-aliasing: {supersample}x supersampling")
        if show_dimensions:
            print(f"  Dimensions: Enabled")
    
    def convert(self, output_format='dxf', output_path=None, 
                views=['front', 'top', 'side']):
        """
        Convert 3D model to 2D CAD drawing
        
        Args:
            output_format: 'dxf', 'svg', or 'png'
            output_path: Custom output path (optional)
            views: List of views to include
        """
        convert_start = time.time()
        
        if output_path is None:
            output_path = self.model_path.with_suffix(f'.{output_format}')
        
        print(f"\nConverting {self.model_path.name} to {output_format.upper()}...")
        
        # --- Projection timing ---
        proj_start = time.time()
        projections = {}
        for view in views:
            view_start = time.time()
            projections[view] = self.get_projection(view)
            view_elapsed = time.time() - view_start
            print(f"  ⏱  Projection '{view}': {view_elapsed:.3f}s")
        proj_time = time.time() - proj_start
        
        # --- Export timing ---
        export_start = time.time()
        
        fmt = output_format.lower()
        
        if fmt == 'dxf':
            self.export_to_dxf_basic(output_path, views)
        elif fmt == 'dxf_full':
            self.export_to_dxf(output_path, views)
        elif fmt == 'dxf_basic':
            self.export_to_dxf_basic(output_path, views)
        elif fmt == 'svg':
            self.export_to_svg(output_path, views)
        elif fmt == 'png':
            self.export_to_png(output_path, views)
        else:
            print(f"[!] Unsupported format: {output_format}")
            print("  Supported formats: dxf, dxf_basic, svg, png")
            return
        
        export_time = time.time() - export_start
        total_time = time.time() - convert_start
        cpu_time = time.process_time() - self._cpu_start
        
        # --- Output file size ---
        file_size = 0
        output_file = Path(output_path)
        if output_file.exists():
            file_size = output_file.stat().st_size
        
        # --- Accuracy Metrics ---
        total_entities = 0
        total_proj_vertices = 0
        total_segment_lengths = []
        bbox_accuracy_scores = []
        
        # 3D model reference extents
        model_extents = self.mesh.extents  # [width, height, depth]
        total_mesh_edges = len(self.mesh.edges_unique)
        
        for view, proj in projections.items():
            if proj is not None and hasattr(proj, 'entities') and hasattr(proj, 'vertices'):
                total_entities += len(proj.entities)
                total_proj_vertices += len(proj.vertices)
                
                # Measure segment lengths (curve resolution)
                for entity in proj.entities:
                    if hasattr(entity, 'points') and len(entity.points) >= 2:
                        pts = proj.vertices[entity.points]
                        seg_len = np.linalg.norm(pts[-1] - pts[0])
                        if seg_len > 0:
                            total_segment_lengths.append(seg_len)
                
                # Bounding box accuracy: compare 2D extents vs expected 3D footprint
                if len(proj.vertices) > 2:
                    proj_min = proj.vertices.min(axis=0)
                    proj_max = proj.vertices.max(axis=0)
                    proj_extents = proj_max - proj_min
                    # Ratio of 2D extent to max 3D extent (ideal ≈ 1.0)
                    model_max = max(model_extents)
                    if model_max > 0:
                        ratio = max(proj_extents) / model_max
                        bbox_accuracy_scores.append(min(ratio, 1.0 / max(ratio, 1e-10)))
        
        # Compute accuracy scores
        edge_detection_rate = (total_entities / max(total_mesh_edges, 1)) * 100
        edge_detection_rate = min(edge_detection_rate, 100.0)
        
        avg_segment_length = np.mean(total_segment_lengths) if total_segment_lengths else 0
        model_scale = max(model_extents) if len(model_extents) > 0 else 1.0
        # Curve resolution: smaller segments relative to model = higher resolution
        curve_resolution = (1.0 - min(avg_segment_length / max(model_scale, 1e-10), 1.0)) * 100 if avg_segment_length > 0 else 0
        
        bbox_accuracy = np.mean(bbox_accuracy_scores) * 100 if bbox_accuracy_scores else 0
        
        # Overall accuracy score (weighted average)
        overall_accuracy = (
            edge_detection_rate * 0.40 +
            curve_resolution * 0.30 +
            bbox_accuracy * 0.30
        )
        
        # --- Performance Summary ---
        print(f"\n{'='*55}")
        print(f"  📊 PERFORMANCE METRICS")
        print(f"{'='*55}")
        print(f"  Model Load Time    : {self._load_time:.3f}s")
        print(f"  Projection Time    : {proj_time:.3f}s  ({len(views)} views)")
        print(f"  Export Time         : {export_time:.3f}s")
        print(f"  Total Time          : {total_time:.3f}s")
        print(f"  CPU Time            : {cpu_time:.3f}s")
        if file_size > 0:
            if file_size >= 1024 * 1024:
                print(f"  Output File Size    : {file_size / (1024*1024):.2f} MB")
            else:
                print(f"  Output File Size    : {file_size / 1024:.2f} KB")
        print(f"  Model Faces         : {len(self.mesh.faces)}")
        print(f"  Model Vertices      : {len(self.mesh.vertices)}")
        print(f"{'='*55}")
        print(f"  🎯 ACCURACY METRICS")
        print(f"{'='*55}")
        print(f"  Edge Detection Rate : {edge_detection_rate:.1f}%  ({total_entities}/{total_mesh_edges} edges)")
        print(f"  Entity Count        : {total_entities}  (across {len(views)} views)")
        print(f"  Projection Vertices : {total_proj_vertices}")
        print(f"  Curve Resolution    : {curve_resolution:.1f}%  (avg segment: {avg_segment_length:.4f})")
        print(f"  Bounding Box Acc.   : {bbox_accuracy:.1f}%")
        print(f"  ── Overall Accuracy : {overall_accuracy:.1f}% ──")
        print(f"{'='*55}")


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Convert 3D models to 2D CAD drawings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 3d_to_2d_cad.py model.stl
  python 3d_to_2d_cad.py model.obj -f svg -o output.svg
  python 3d_to_2d_cad.py model.stl -v front top side isometric
        """
    )
    
    parser.add_argument('input', help='Input 3D model file (STL, OBJ, PLY, STEP/STP, etc.)')
    parser.add_argument('-f', '--format', default='dxf',
                       choices=['dxf', 'dxf_basic', 'dxf_full', 'svg', 'png'],
                       help='Output format (default: dxf basic for ShareCAD). Use dxf_full for layered/dimensioned DXF.')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('-v', '--views', nargs='+',
                       default=['front', 'top', 'side'],
                       choices=['front', 'back', 'left', 'right', 'top', 'bottom', 'side', 'isometric',
                               'section_front', 'section_top', 'section_side'],
                       help='Views to include (default: front top side). Use section_* for section views showing internal features.')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"[!] Error: File not found: {args.input}")
        sys.exit(1)
    
    # Create converter and process
    converter = Model3Dto2DConverter(args.input)
    converter.convert(
        output_format=args.format,
        output_path=args.output,
        views=args.views
    )
    
    print("\n[+] Conversion complete!")


if __name__ == '__main__':
    main()
