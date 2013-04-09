varying vec2 texpos;

void main() {
	gl_Position = gl_Vertex;
	texpos = gl_Vertex.xy*0.5 + 0.5;
}
