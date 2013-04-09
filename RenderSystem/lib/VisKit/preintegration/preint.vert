varying float startpos;
varying float endpos;
void main() {
	gl_Position = gl_Vertex;
	startpos = gl_Vertex.x*0.5 + 0.5;
	endpos = gl_Vertex.y*0.5 + 0.5;
}
