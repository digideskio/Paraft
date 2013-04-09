uniform sampler2D fbo;

varying vec2 texpos;

void main() {
	gl_FragColor = texture2D(fbo, texpos);
}
