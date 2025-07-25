module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        hospital: {
          blue: '#2563eb', // Hospital blue
          light: '#e0f2fe',
          dark: '#1e293b',
        },
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui'],
      },
      boxShadow: {
        soft: '0 4px 24px 0 rgba(37, 99, 235, 0.08)',
      },
    },
  },
  plugins: [],
} 