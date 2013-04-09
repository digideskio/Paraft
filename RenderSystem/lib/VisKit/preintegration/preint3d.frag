uniform sampler1D tf;
uniform float width;
uniform float steps;
uniform vec4 specular;
uniform vec4 diffuse;
uniform float basesteps;

varying float startpos;
varying float endpos;

uniform float distscale;
uniform float deltascale;
uniform float lawldist;
uniform float minstep;


void main() {

	//float t1, t2 = gl_FragCoord.x;
	//float t2 = gl_FragCoord.y;
	float stepadjust;// = t1 != t2 ? basesteps/(steps*(abs(t1 - t2))) : basesteps/steps;
	vec4 color = vec4(0.0);
	vec4 front = vec4(0.0);
	vec4 back = vec4(0.0);
	vec4 cc;
	float t1, t2;

//	float startpos = (gl_FragCoord.x + 0.5)/width;
//	float endpos = (gl_FragCoord.y + 0.5)/width;

	//float i = t1;
	
	float delta = 1.0/width;
	
	float f = 1.0, b = 0.0;
	float scaledmin = minstep/(distscale/max(1.0,abs(gl_FragCoord.x - gl_FragCoord.y)))/width;
	float tprev = -delta;
	float dir = sign(endpos - startpos);
	float dist = abs(startpos - endpos);
	stepadjust = distscale/(abs(gl_FragCoord.x - gl_FragCoord.y));
	t2 = dist;
	t1 = 0.0;
//	t2 = dist;
//	t1 = delta;

	if(distscale == 0.0) {
		gl_FragData[0] = gl_FragData[1] = gl_FragData[2] = vec4(0.0);
	}

	if(floor(gl_FragCoord.x + 0.5) == floor(gl_FragCoord.y + 0.5)) {
		color = texture1D(tf, startpos);
		color.a = 1.0 - pow(1.0 - color.a, distscale);
		color.rgb *= color.a;
		front = color;
		back = vec4(0.0);
	//	front = back = color = vec4(ceil(color.a));
	//	front.rgb = 1.0;
	}
	else {
		
		while(t1 < t2) {
			f = 1.0 - t1/t2;
			b = 0.0 - f;
			cc = texture1D(tf, (t1*dir + startpos));
			cc.a = 1.0 - pow(1.0 - cc.a, stepadjust);
			tprev = t1;
			cc.rgb *= cc.a;
			cc *= (1.0 - color.a);
			front += f*cc;
			back += b*cc;
			color += cc;
			t1 += delta;
			//t1 = min(t1 + delta, t2);
		};
	/*	

		cc = texture1D(tf, endpos);
		cc.a = 1.0 - pow(1.0 - cc.a, stepadjust*0.5);
		cc.a = (1.0 - color.a)*cc.a;
		cc.rgb *= cc.a;
		front += f*cc;
		back += b*cc;
		color += cc; */

	}
	gl_FragData[0] = color;//vec4(startpos, endpos, 0.0, 1.0);;
	gl_FragData[1] = back;
	gl_FragData[2] = front;

	//gl_FragData[0] = gl_FragData[1] = gl_FragData[2] = vec4(1.0);
}
