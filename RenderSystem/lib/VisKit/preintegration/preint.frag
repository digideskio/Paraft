uniform sampler1D tf;
uniform float width;
uniform float steps;
uniform vec4 specular;
uniform vec4 diffuse;
uniform float basesteps;
uniform float distscale;
//varying float startpos;
//varying float endpos;

void main() {
	float t1, t2;// = gl_FragCoord.x;
	//float t2 = gl_FragCoord.y;
	float stepadjust;// = t1 != t2 ? basesteps/(steps*(abs(t1 - t2))) : basesteps/steps;
	vec4 color = vec4(0.0);
	vec4 front = vec4(0.0);
	vec4 back = vec4(0.0);
	vec4 cc;
	
	//float i = t1;
	
	float delta = 1.0/width;

	float startpos = (gl_FragCoord.x + 0.5)/width;
	float endpos = (gl_FragCoord.y + 0.5)/width;

	
	float f, b;
	float dir = sign(gl_FragCoord.y - gl_FragCoord.x);
	float dist = (abs(gl_FragCoord.x - gl_FragCoord.y) + 1.0)/width;
	stepadjust = distscale*basesteps/(steps*dist*width);
	t2 = dist;
	t1 = delta;


	if(dist <= delta) {
		color = texture1D(tf, endpos);
		color.a = 1.0 - pow(1.0 - color.a, stepadjust);
		color.rgb *= color.a;
		front = vec4(0.0);
		back = color;
	}
	else {
		do {
			cc = texture1D(tf, (t1*dir + startpos));
			f = t1/t2;
			b = 1.0 - f;
			cc.a = 1.0 - pow(1.0 - cc.a, stepadjust);
			//cc.a = 1.0 - pow(1.0 - cc.a, dist);
			cc.a = (1.0 - color.a)*cc.a;
			cc.rgb *= cc.a;
			front += f*cc;
			back += b*cc;
			color += cc;
			t1 += delta;
		} while (t1 <= t2);
	}
/*

	do {
		cc = texture1D(tf, (i + 0.5)/width);
		f = t1 == t2 ? 0.5 : abs(t2 - i)/abs(t2 - t1);
		b = 1.0 - f;
		cc.a = (1.0 - color.a)*(1.0 - pow(1.0 - cc.a, stepadjust));
		color.rgb += cc.a*cc.rgb;
		front.rgb += cc.a*f*cc.rgb*diffuse.rgb;
		back.rgb += cc.a*b*cc.rgb*diffuse.rgb;
		color.a += cc.a;
		front.a += cc.a*f;
		back.a += cc.a*b;
		i += (t1 <= t2 ? 1.0 : -1.0);
	} while(
			((t1 <= t2 && i < t2) || (t1 > t2 && i > t2))); */
	//color.rgb = color.a;
	//color = vec4(vec3(stepadjust),1.0);
	gl_FragData[0] = color;//vec4(startpos, endpos, 0.0, 1.0);;
	gl_FragData[1] = back;
	gl_FragData[2] = front;
}
